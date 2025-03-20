/* Basic configuration header for guac-kalman-filter */
#ifndef _GUAC_KALMAN_CONFIG_H
#define _GUAC_KALMAN_CONFIG_H

/* Version information */
#define PACKAGE_VERSION "0.1.0"
#define PACKAGE_STRING "guac-kalman-filter 0.1.0"
#define PACKAGE_NAME "guac-kalman-filter"

/* System-specific configurations */
#define HAVE_NANOSLEEP 1
#define HAVE_CLOCK_GETTIME 1

/* Protocol settings */
#define GUACD_DEFAULT_PORT 4822
#define GUACD_LOG_INFO 4
#define GUACD_LOG_ERROR 2
#define GUACD_LOG_DEBUG 8

#endif /* _GUAC_KALMAN_CONFIG_H */ 
