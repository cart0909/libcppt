#include <omp.h>
#include <iostream>
#include <cstdio>
#include <gtest/gtest.h>
#include "basic_datatype/tic_toc.h"

TEST(openmp, parallel_barrier) {
    int thread_id;
    printf("I am the main thread.\n\n");
    omp_set_num_threads(4);
    #pragma omp parallel private(thread_id)
    {
        thread_id = omp_get_thread_num();

        printf("Thread %d: Hello.\n", thread_id);
        #pragma omp barrier
        printf("Thread %d: Bye bye.\n", thread_id);
    }

    printf("\nI am the main thread.\n\n");
}

TEST(openmp, loop_consturct) {
    int a[12];
    int i, tid, nthreads;

    #pragma omp parallel for private(tid)
    for(i=0; i<12; i++){
        a[i] = i;
        tid = omp_get_thread_num();
        printf("Thread %d: a[%d] = %d\n", tid, i, a[i]);
    }
}

void calcSum(int n){
    int tid = omp_get_thread_num();
    int value = 0;
    for (int i=0; i<=n; i++) {
        usleep(1000);
        value += i;
    }
    printf("Thread %d: sum of %d = %d\n", tid, n, value);
}

TEST(openmp, section) {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            calcSum(3);
        }
        #pragma omp section
        {
            calcSum(5);
        }
        #pragma omp section
        {
            calcSum(7);
        }
        #pragma omp section
        {
            calcSum(9);
        }
    }
}

TEST(openmp, single) {
    int tid;
    int a[12];
    for (int i=0; i<12; i++){
        a[i] = i;
    }

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();

        #pragma omp single
        {
            for (int i=0; i<12; i++) {
                a[i] = 0;
            }
            printf("---TID %d: a[i] has been initialized.\n", tid);
        }

        #pragma omp for
        for(int i=0; i<12; i++){
            printf("TID %d: a[%d] = %d\n", tid, i, a[i]);
        }

    }
}

TEST(openmp, master){
    int tid;
    int a[12];
    for (int i=0; i<12; i++){
        a[i] = i;
    }

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();

        #pragma omp master
        {
            usleep(300);
            for (int i=0; i<12; i++) {
                a[i] = 0;
            }
            printf("---TID %d: a[i] has been initialized.\n", tid);
        }

        #pragma omp for
        for(int i=0; i<12; i++){
            printf("TID %d: a[%d] = %d\n", tid, i, a[i]);
        }

    }
}

TEST(openmp, simd) {
    int N = 50000;
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];
    float sum = 0.0f;
    {
        for(int i = 0; i < N; ++i) {
            a[i] = (i + 1);
            b[i] = (i + 1) * 0.3f;
            c[i] = 0;
        }
    }

    TicToc tic;
    {
    for(int i = 0; i < N; ++i) {
        c[i] = a[i] + 2.0f * b[i];
        sum += c[i];
    }
    std::cout << "g++    " << tic.toc() << " (ms) " << sum << std::endl;
    sum = 0.0f;
    tic.tic();

#pragma omp parallel for simd reduction(+:sum)
    for(int i = 0; i < N; ++i) {
        c[i] = a[i] + 2.0f * b[i];
        sum += c[i];
    }
    std::cout << "openmp " << tic.toc() << " (ms) " << sum << std::endl;
    }
    delete[] a;
    delete[] b;
}
