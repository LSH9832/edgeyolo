/*
 * print_utils.h
 *
 *  Created on: Mar 6, 2024
 *      Author: LSH9832
 */
#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H

#define YELLOW "\033[33m" /* Yellow */
#define GREEN "\033[32m"  /* Green  */
#define RED "\033[31m"  /* Red  */
#define END "\033[0m"
#define ENDL "\033[0m" << std::endl

#define WARN (std::cout << YELLOW)
#define INFO (std::cout << GREEN)
#define ERROR (std::cout << RED)

#include <sys/time.h>

static timeval get_now_time() {
    timeval _t;
    gettimeofday(&_t, NULL);
    return _t;
}

static int get_time_interval(timeval _t1, timeval _t2) {
  return (int)((_t2.tv_sec - _t1.tv_sec) * 1000 + (_t2.tv_usec - _t1.tv_usec) / 1000);
}

static float get_time_interval_f(timeval _t1, timeval _t2) {
  return ((float)(_t2.tv_sec - _t1.tv_sec) * 1000. + (float)(_t2.tv_usec - _t1.tv_usec) / 1000.);
}

class TimeCount {
  std::vector<timeval> tics;
  std::vector<timeval> tocs;

public:
  size_t length() {
    return tics.size();
  }

  void tic(int idx) {
    timeval now_time = get_now_time();
    while (idx >= this->length()) {
      tics.push_back(now_time);
      tocs.push_back(now_time);
    }
    tics[idx] = now_time;
    tocs[idx] = now_time;
  }

  int get_timeval(int idx) {
    idx = MIN(idx, this->length()-1);
    return get_time_interval(tics[idx], tocs[idx]);
  }

  float get_timeval_f(int idx) {
    idx = MIN(idx, this->length()-1);
    return get_time_interval_f(tics[idx], tocs[idx]);
  }

  int toc(int idx) {
    idx = MIN(idx, this->length()-1);
    tocs[idx] = get_now_time();
    return this->get_timeval(idx);
  }

};

#endif
