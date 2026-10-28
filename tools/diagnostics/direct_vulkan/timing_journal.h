// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <stdexcept>
#include <string>
// Exclusive incremental sample journal; directory durability and failure receipts
// are the parent runner's responsibilities.
class SinterTimingJournal {
public:
 explicit SinterTimingJournal(const char* path){fd_=open(path,O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);if(fd_<0)throw std::runtime_error("timing journal create");}
 SinterTimingJournal(const SinterTimingJournal&)=delete;SinterTimingJournal& operator=(const SinterTimingJournal&)=delete;
 ~SinterTimingJournal(){if(fd_>=0)close(fd_);}
 void append(const std::string& s){const char* p=s.data();size_t left=s.size();while(left){ssize_t n=write(fd_,p,left);if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("timing journal write");p+=n;left-=n;}if(fsync(fd_))throw std::runtime_error("timing journal sync");}
 void finish(){if(fd_<0||fsync(fd_))throw std::runtime_error("timing journal final sync");int f=fd_;fd_=-1;if(close(f))throw std::runtime_error("timing journal close");}
private:int fd_=-1;
};
