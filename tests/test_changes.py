import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('check_changes',Path(__file__).resolve().parents[1]/'tools/check_changes.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

@pytest.mark.parametrize('old,new,kind', [('0.1.0','0.1.1','breaking'),('1.1.0','1.2.0','breaking'),('0.1.0','0.1.0','compatible'),('0.1.0','0.1.1','initial')])
def test_incompatible_or_unversioned_change_is_rejected(old,new,kind):
    with pytest.raises(ValueError): c.validate_transition(old,new,kind)

@pytest.mark.parametrize('old,new,kind', [('0.1.0','0.2.0','breaking'),('1.1.0','2.0.0','breaking'),('0.1.0','0.1.1','compatible')])
def test_valid_transitions(old,new,kind): c.validate_transition(old,new,kind)
