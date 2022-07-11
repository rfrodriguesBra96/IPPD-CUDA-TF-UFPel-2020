

#include "nbody.hpp"
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <sys/time.h>
#include "cuda_runtime_api.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <cassert>
#include <ctime>

using namespace std;

__global__ void nbody(Particle* d_particles, Particle *output,int number_of_particles,float time_interval){
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(id < number_of_particles) {  
        
    Particle* this_particle = &output[id];
    
    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
    float total_force_x = 0.0f, total_force_y = 0.0f, total_force_z = 0.0f;
    int i;

    for(i = 0; i < number_of_particles; i++) {
                
      if(i != id) {
                                             
        Particle* this_particle1 = d_particles + id;
        Particle* this_particle2 = d_particles + i;
        float* force_x_aux = &force_x; 
        float* force_y_aux = &force_y; 
        float* force_z_aux = &force_z;                          

        float difference_x, difference_y, difference_z;
        float distance_squared, distance;
        float force_magnitude;

        difference_x = this_particle2->position_x - this_particle1->position_x;
        difference_y = this_particle2->position_y - this_particle1->position_y;
        difference_z = this_particle2->position_z - this_particle1->position_z;

        distance_squared = difference_x * difference_x +
                           difference_y * difference_y +
                           difference_z * difference_z;

        distance = std::sqrt(distance_squared);//sqrtf(distance_squared);

        force_magnitude = GRAVITATIONAL_CONSTANT * (this_particle1->mass) * (this_particle2->mass) / distance_squared;

        *force_x_aux = (force_magnitude / distance) * difference_x;
        *force_y_aux = (force_magnitude / distance) * difference_y;
        *force_z_aux = (force_magnitude / distance) * difference_z;
        

        total_force_x += force_x;
        total_force_y += force_y;
        total_force_z += force_z;
      }                        
    }
       
        float velocity_change_x, velocity_change_y, velocity_change_z;
        float position_change_x, position_change_y, position_change_z;
        
        this_particle->mass = d_particles[id].mass;
            
        velocity_change_x = total_force_x * (time_interval / this_particle->mass);
        velocity_change_y = total_force_y * (time_interval / this_particle->mass);
        velocity_change_z = total_force_z * (time_interval / this_particle->mass);
            
        position_change_x = d_particles[id].velocity_x + velocity_change_x * (0.5 * time_interval);
        position_change_y = d_particles[id].velocity_y + velocity_change_y * (0.5 * time_interval);
        position_change_z = d_particles[id].velocity_z + velocity_change_z * (0.5 * time_interval);
            
        this_particle->velocity_x = d_particles[id].velocity_x + velocity_change_x;
        this_particle->velocity_y = d_particles[id].velocity_y + velocity_change_y;
        this_particle->velocity_z = d_particles[id].velocity_z + velocity_change_z;
        
        this_particle->position_x = d_particles[id].position_x + position_change_x;
        this_particle->position_y = d_particles[id].position_y + position_change_y;
        this_particle->position_z = d_particles[id].position_z + position_change_z;

  }

}

int main (int argc, char** argv) { 
    if(argc < 2) {
        std::cout << "Informe um arquivo com os parâmetros de entrada: ./nbody_simulation <input_file.in>\n";
        std::abort();
    }
   int n = atoi(argv[2]);

    
    Particle* particle_array  = nullptr;
    Particle* particle_array2 = nullptr;
    Particle* dev_particle_array;
    Particle* dev_particle_array2; 
    
    FILE *input_data = fopen(argv[1], "r");
    Particle_input_arguments(input_data);

    particle_array  = Particle_array_construct(number_of_particles);
  particle_array2 = Particle_array_construct(number_of_particles);

    Particle_array_initialize(particle_array, number_of_particles);
    
  printf("\nProcessando simulação NBody....\n");
    
    
  long start = wtime();   
    
  cudaMalloc((void**)&dev_particle_array, number_of_particles * sizeof(Particle));
  cudaMalloc((void**)&dev_particle_array2, number_of_particles * sizeof(Particle));
    
  cudaMemcpy(dev_particle_array, particle_array, number_of_particles *sizeof(Particle),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_particle_array2, particle_array2, number_of_particles *sizeof(Particle),cudaMemcpyHostToDevice);
   
  //!-------KERNEL-----
  
  for(int timestep = 1; timestep <= number_of_timesteps; timestep++) {
     
    nbody<<<number_of_particles,n>>>(dev_particle_array,dev_particle_array2,number_of_particles,time_interval);
        
    /* swap arrays */
    Particle * tmp = particle_array;
    particle_array = particle_array2;
    particle_array2 = tmp;
        
    //printf("   Iteração %d OK\n", timestep);
        
    cudaError err = cudaMemcpy(particle_array, dev_particle_array, number_of_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
      printf("CUDA error ao copiar dados para o Host: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
   printf(" iteração: %d OK\n ",timestep);
  }
        
  cudaFree(dev_particle_array);
  cudaFree(dev_particle_array2);

    long end = wtime();
    double time = (end - start) / 1000000.0;

   
    printf("\nSimulação NBody executada com sucesso.\n");
    
    printf("Numero de threads: %d\n",n);
    printf("Nro. de Iterações : %d\n", number_of_timesteps);
    printf("Nro. de Partículas: %d\n", number_of_particles);
    printf("Tempo: %.8f segundos\n", time);


    // #ifdef VERBOSE
        //Imprimir saída para arquivo
        printf("\nImprimindo saída em arquivo...\n");
        FILE * fileptr = fopen("nbody_simulation.out", "w");
        Particle_array_output_xyz(fileptr, particle_array, number_of_particles);
        printf("Saída da simulação salva no arquivo nbody_simulation.out\n");
    // #endif
                 
        particle_array  = Particle_array_destruct(particle_array, number_of_particles);
        particle_array2 = Particle_array_destruct(particle_array2, number_of_particles);
  
      
    
    return PROGRAM_SUCCESS_CODE;
  }