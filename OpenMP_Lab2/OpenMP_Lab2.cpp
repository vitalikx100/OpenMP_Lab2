#include <omp.h>
#include "stdio.h"
#include "stdlib.h"
#include "locale.h"
#define Q 28
#define NMAX 3200000

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");

    printf("Вариант 19: double, 7, 3 200 000, [4, 8, 16], 28\n\n");

    omp_set_num_threads(16);

    double* mass_1 = (double*)malloc(NMAX * sizeof(double));
    double* mass_2 = (double*)malloc(NMAX * sizeof(double));
    double* mass_3 = (double*)malloc(NMAX * sizeof(double));
    double* mass_4 = (double*)malloc(NMAX * sizeof(double));
    double* mass_5 = (double*)malloc(NMAX * sizeof(double));
    double* mass_6 = (double*)malloc(NMAX * sizeof(double));
    double* mass_7 = (double*)malloc(NMAX * sizeof(double));

    for (int i = 0; i < NMAX; i++) {
        mass_1[i] = 1.0;
        mass_2[i] = 1.0;
        mass_3[i] = 1.0;
        mass_4[i] = 1.0;
        mass_5[i] = 1.0;
        mass_6[i] = 1.0;
        mass_7[i] = 1.0;
    }

    double sum_sequence, sum_reduction, sum_critical, sum_atomic;
    double t_s = 0.0, t_p = 0.0, t_c = 0.0, t_a = 0.0, t_r = 0.0;
    double start_time_sequence, start_time_parallel, start_time_reduction, start_time_critical, start_time_atomic;
    const int reps = 20;
    int i, j;

    for (int rep = 0; rep < reps; rep++)
    {
        sum_sequence = 0, sum_reduction = 0, sum_critical = 0, sum_atomic = 0;

        start_time_sequence = omp_get_wtime();
        for (i = 0; i < NMAX; i++)
        {
            for (j = 0; j < Q; j++)
            {
                sum_sequence += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
            }
        }
        t_s = omp_get_wtime() - start_time_sequence;


        start_time_parallel = omp_get_wtime();
#pragma omp parallel private(i, j)
        {
            t_p = omp_get_wtime() - start_time_parallel;

            start_time_reduction = omp_get_wtime();
#pragma omp for reduction(+:sum_reduction)
            for (i = 0; i < NMAX; i++)
            {
                for (j = 0; j < Q; j++)
                {
                    sum_reduction += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
                }
            }
            t_r = omp_get_wtime() - start_time_reduction;

            start_time_critical = omp_get_wtime();
#pragma omp for
            for (i = 0; i < NMAX; i++)
            {
                for (j = 0; j < Q; j++)
                {
                    double local_sum = mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
#pragma omp critical
                    {
                        sum_critical += local_sum;
                    }
                }
            }
            t_c = omp_get_wtime() - start_time_critical;

            start_time_atomic = omp_get_wtime();
#pragma omp for
            for (i = 0; i < NMAX; i++)
            {
                for (j = 0; j < Q; j++)
                {
                    double local_sum = mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
#pragma omp atomic
                    sum_atomic += local_sum;
                }
            }
            t_a = omp_get_wtime() - start_time_atomic;
        }
    }
    

    printf("Среднее время выполнения:\n");
    printf("Последовательный алгоритм: %f секунд. Итоговая сумма %.2f\n", t_s, sum_sequence / Q);
    printf("Инициализация параллельной области: %f секунд\n", t_p);
    printf("Параллельный алгоритм с reduction: %f секунд. Итоговая сумма %.2f\n", t_r, sum_reduction / Q);
    printf("Параллельный алгоритм с critical: %f секунд. Итоговая сумма %.2f\n", t_c, sum_critical / Q);
    printf("Параллельный алгоритм с atomic: %f секунд. Итоговая сумма %.2f\n\n", t_a, sum_atomic / Q);

    double a_r = t_s / t_r; double a_c = t_s / t_c; double a_a = t_s / t_a;
    double a_rp = t_s / (t_r + t_p); double a_cp = t_s / (t_c + t_p); double a_ap = t_s / (t_a + t_p);

    printf("Вычисление итогового ускорения алгоритмов:\n");
    printf("1. Без учета инициализации параллельной области:\n");
    printf("Reduction a_r: %f \n", a_r);
    printf("Critical a_c: %f \n", a_c);
    printf("Atomic a_a: %f \n\n", a_a);

    printf("2. С учетом инициализации параллельной области:\n");
    printf("Reduction a_rp: %f \n", a_rp);
    printf("Critical a_cp: %f \n", a_cp);
    printf("Atomic a_ap: %f \n", a_ap);


    free(mass_1);
    free(mass_2);
    free(mass_3);
    free(mass_4);
    free(mass_5);
    free(mass_6);
    free(mass_7);

    return 0;
}