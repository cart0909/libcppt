#include "thread_pool_mgr.h"

ThreadPool& ThreadPoolMgr::Pool()
{
    static ThreadPool pool(1);
    return pool;
}
