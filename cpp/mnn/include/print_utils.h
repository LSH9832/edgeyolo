#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H

#define YELLOW "\033[33m" /* Yellow */
#define GREEN "\033[32m"  /* Green  */
#define RED "\033[31m"  /* Red  */
#define ENDL "\033[0m" << std::endl

#define WARN (std::cout << YELLOW)
#define INFO (std::cout << GREEN)
#define ERROR (std::cout << RED)

timeval get_now_time() {
    timeval _t;
    gettimeofday(&_t, NULL);
    return _t;
}

int get_time_interval(timeval _t1, timeval _t2) {
  return (int)((_t2.tv_sec - _t1.tv_sec) * 1000 + (_t2.tv_usec - _t1.tv_usec) / 1000);
}

#endif