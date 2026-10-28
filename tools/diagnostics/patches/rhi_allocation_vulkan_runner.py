# SPDX-License-Identifier: Apache-2.0
"""Fail-closed diagnostic runner scaffold; actual Vulkan execution is blocked.

Current experimental RHI does not expose the instance dispatch provider. Do not
replace that absence with a library name, vkGetInstanceProcAddr from a separate
loader, or a caller assertion. No GPU is created until the source-backed provider
and full immutable runtime/ICD closure checks are implemented and reviewed.
"""
import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from benchmarks.depth_pyramid_timestamp_probe import BorrowedLease, LOCK


class IdentityUnavailable(RuntimeError):
    pass


@contextmanager
def owned_lease():
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = BorrowedLease(fd)
        lease.verify()
        yield lease
    finally:
        os.close(fd)


def verify_live_runtime_and_provider():
    """Intentionally blocks: no backend-observed provider receipt exists yet."""
    raise IdentityUnavailable('RHI instance dispatch provider and immutable loaded runtime/ICD closure are not exposed or verified')


def write_once(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial receipt stays visible and cannot be mistaken for a valid JSON receipt.
        raise


def run(*, uuid, driver, output, preflight=verify_live_runtime_and_provider,
        create_device=None, run_cases=None, lease_context=owned_lease):
    """Control flow for a future independently reviewed identity verifier.

    No callable is supplied for device creation in production. Injection is for
    CPU orchestration tests, not an authorization or evidence interface.
    """
    if not isinstance(uuid, bytes) or len(uuid) != 16 or not any(uuid): raise ValueError('nonzero UUID required')
    if type(driver) is not int or driver < 0: raise ValueError('raw driver version required')
    path = Path(output)
    if path.exists() or path.is_symlink(): raise FileExistsError('output already exists')
    error = None
    result = None
    try:
        with lease_context() as lease:
            device = None
            cases_started = False
            failures = []
            try:
                lease.verify()
                preflight()  # Must fail until actual provider and closure proof exists.
                if create_device is None or run_cases is None:
                    raise IdentityUnavailable('reviewed device and case bindings are unavailable')
                device = create_device()
                # Contract: the case function owns device.close() in its own finally.
                cases_started = True
                result = run_cases(device, lease=lease, expected_uuid=uuid, expected_driver=driver)
            except Exception as exc:
                failures.append(exc)
            finally:
                if device is not None and not cases_started:
                    try:
                        device.close()
                    except Exception as exc:
                        failures.append(exc)
            try:
                lease.verify()  # After teardown, before releasing the lock.
            except Exception as exc:
                failures.append(exc)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise ExceptionGroup('diagnostic and teardown/lease failures', failures)
    except Exception as exc:
        error = exc
    if error is not None:
        write_once(path, {'status': 'failure', 'error_type': type(error).__name__, 'error': str(error),
                          'scope': 'diagnostic only; no device or allocation qualification'})
        raise error
    write_once(path, {'status': 'diagnostic_only', 'result': result,
                      'scope': 'no physical backing or performance qualification'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uuid', required=True, help='expected physical UUID in hex')
    parser.add_argument('--driver', required=True, type=int, help='expected raw Vulkan driver version')
    parser.add_argument('--output', required=True, help='new write-once JSON receipt')
    args = parser.parse_args()
    run(uuid=bytes.fromhex(args.uuid.replace('-', '')), driver=args.driver, output=args.output)


if __name__ == '__main__': main()
