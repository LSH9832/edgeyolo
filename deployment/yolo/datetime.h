#pragma once
#ifndef PYLIKE_DATETIME_H
#define PYLIKE_DATETIME_H

#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include "./str.h"
#include <time.h>
#include <sys/time.h>


namespace pytime {

    static double time() {
        timeval _t;
        gettimeofday(&_t, NULL);
        double ret = (double)_t.tv_sec + (double)_t.tv_usec / 1000000.0;
        return ret;
    }

    static void sleep(double sec) {
        double t0 = pytime::time();
        while (1) {
            if (pytime::time()-t0 > sec) break;
        }
    }

}


class TimeCount {
  std::vector<double> tics;
  std::vector<double> tocs;

public:
  size_t length() {
    return tics.size();
  }

  void tic(int idx) {
    double now_time = pytime::time();
    while (idx >= this->length()) {
      tics.push_back(now_time);
      tocs.push_back(now_time);
    }
    tics[idx] = now_time;
    tocs[idx] = now_time;
  }

  int get_timeval(int idx) {
    idx = MIN(idx, this->length()-1);
    return 1000 * (tocs[idx] - tics[idx]);
  }

  double get_timeval_f(int idx) {
    idx = MIN(idx, this->length()-1);
    return 1000.0 * (tocs[idx] - tics[idx]);
  }

  int toc(int idx) {
    idx = MIN(idx, this->length()-1);
    tocs[idx] = pytime::time();
    return this->get_timeval(idx);
  }
  
};


namespace datetime {

    class Datetime {
    public:
        // 构造函数，默认当前时间
        Datetime() {
            now();
        }

        Datetime now() {
            tp_ = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(tp_);
            tm_ = *std::localtime(&now_c);
            return *this;
        }

        pystring strftime(pystring format="") {
            if (!format.length() || format.empty()) {
                format = "%Y-%m-%d %H:%M:%S.%ms";
            }
            std::ostringstream oss, oss_ms;
            auto duration = tp_.time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            oss_ms << std::setfill('0') << std::setw(3) << millis % 1000;
            format = format.replace("%ms", oss_ms.str());
            oss << std::put_time(&tm_, format.c_str());
            return pystring(oss.str());
        }

        int year() const {
            return tm_.tm_year + 1900;
        }

        int month() const {
            return tm_.tm_mon + 1;
        }

        int day() const {
            return tm_.tm_mday;
        }

        int hour() const {
            return tm_.tm_hour;
        }

        int minute() const {
            return tm_.tm_min;
        }

        int second() const {
            return tm_.tm_sec;
        }

    private:
        std::tm tm_;
        std::chrono::system_clock::time_point tp_;
    };


    namespace datetime {

        static Datetime now() {
            return Datetime();
        }
    }

}


#endif