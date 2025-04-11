/* AEP 4380 C++ Final 
 *
 * Modelling Particle-Mesh N-body gravity
 *
 * Run on an M1 Macbook Air with Apple clang 15.0.0
 *
 * Farhan Naqib 12/19/2024
 */

#include <cstdlib>
#include <cmath>

#include <iostream> // stream IO
#include <fstream> // stream file IO
#include <sstream> // string streams
#include <iomanip> // to format the output
#include <string> // STD strings
#include <vector> // STD vector class

#include <fftw3.h>

using namespace std;

const int Nx = 128, Ny = 128, Nz = 1;
const double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
const double G = 1.0;
const double dt = 0.01;
const int n_particles = 2;

struct Particle {
    double x, y, z;
    double vx, vy, vz;
};

void map_to_grid(const std::vector<Particle>& particles, std::vector<double>& density, bool useCIC) {
    std::fill(density.begin(), density.end(), 0.0);
    double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;

    for (const auto& p : particles) {
        double gx = p.x / dx;
        double gy = p.y / dy;
        double gz = p.z / dz;

        if (useCIC) {
            // CIC scheme
            int ix0 = static_cast<int>(std::floor(gx)) % Nx;
            int iy0 = static_cast<int>(std::floor(gy)) % Ny;
            int iz0 = static_cast<int>(std::floor(gz)) % Nz;
            int ix1 = (ix0 + 1) % Nx;
            int iy1 = (iy0 + 1) % Ny;
            int iz1 = (iz0 + 1) % Nz;

            double wx1 = gx - ix0;
            double wx0 = 1.0 - wx1;
            double wy1 = gy - iy0;
            double wy0 = 1.0 - wy1;
            double wz1 = gz - iz0;
            double wz0 = 1.0 - wz1;

            density[iz0 + Nz * (iy0 + Ny * ix0)] += wx0 * wy0 * wz0;
            density[iz0 + Nz * (iy0 + Ny * ix1)] += wx1 * wy0 * wz0;
            density[iz0 + Nz * (iy1 + Ny * ix0)] += wx0 * wy1 * wz0;
            density[iz0 + Nz * (iy1 + Ny * ix1)] += wx1 * wy1 * wz0;

            density[iz1 + Nz * (iy0 + Ny * ix0)] += wx0 * wy0 * wz1;
            density[iz1 + Nz * (iy0 + Ny * ix1)] += wx1 * wy0 * wz1;
            density[iz1 + Nz * (iy1 + Ny * ix0)] += wx0 * wy1 * wz1;
            density[iz1 + Nz * (iy1 + Ny * ix1)] += wx1 * wy1 * wz1;
        } else {
            // NGP scheme
            int ix = static_cast<int>(std::round(gx)) % Nx;
            int iy = static_cast<int>(std::round(gy)) % Ny;
            int iz = static_cast<int>(std::round(gz)) % Nz;

            density[iz + Nz * (iy + Ny * ix)] += 1.0;
        }
    }
}

// Solve Poisson's equation using FFT
void solve_poisson_3d(std::vector<double>& density, std::vector<double>& potential) {
    // need to cut last dimension by half
    int Nzh = Nz / 2 + 1;
    int N = Nx * Ny * Nzh;
    fftw_complex *rho_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *phi_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    fftw_plan forward_plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, density.data(), rho_k, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, phi_k, potential.data(), FFTW_ESTIMATE);

    fftw_execute_dft_r2c(forward_plan, density.data(), rho_k);

    // Solve Poisson equation in Fourier space
    for (int ix = 0; ix < Nx; ++ix) {
        double kx = (ix < Nx / 2) ? ix : ix - Nx;
        kx *= 2 * M_PI / Lx;
        for (int iy = 0; iy < Ny; ++iy) {
            double ky = (iy < Ny / 2) ? iy : iy - Ny;
            ky *= 2 * M_PI / Ly;
            for (int iz = 0; iz < Nzh; ++iz) {
                double kz = (iz < Nz / 2 ? iz : iz - Nz) * 2 * M_PI / Lz;

                int idx = iz + Nzh * (iy + Ny * ix);
                double k2 = (kx * kx) + (ky * ky) + (kz * kz); 
                if (k2 == 0) {
                    phi_k[idx][0] = 0;
                    phi_k[idx][1] = 0;
                } 
                else {
                    phi_k[idx][0] = -rho_k[idx][0] / k2;
                    phi_k[idx][1] = -rho_k[idx][1] / k2;
                }
            }
        }
    }

    fftw_execute_dft_c2r(backward_plan, phi_k, potential.data());

    N = Nx * Ny * Nz;
    for (int i = 0; i < N; ++i) {
        potential[i] /= N;
    }

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(rho_k);
    fftw_free(phi_k);
}

void compute_acceleration(const std::vector<double>& potential, std::vector<Particle>& particles, bool useCIC) {
    double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;

    for (auto& p : particles) {
        double gx = p.x / dx;
        double gy = p.y / dy;
        double gz = p.z / dz;

        int ix = static_cast<int>(std::floor(gx)) % Nx;
        int iy = static_cast<int>(std::floor(gy)) % Ny;
        int iz = static_cast<int>(std::floor(gz)) % Nz;

        if (useCIC) {
            // CIC acceleration computation
            int ix0 = ix, ix1 = (ix + 1) % Nx;
            int iy0 = iy, iy1 = (iy + 1) % Ny;
            int iz0 = iz, iz1 = (iz + 1) % Nz;

            double wx1 = gx - ix0;
            double wx0 = 1.0 - wx1;
            double wy1 = gy - iy0;
            double wy0 = 1.0 - wy1;
            double wz1 = gz - iz0;
            double wz0 = 1.0 - wz1;

            double ax = 0.0, ay = 0.0, az = 0.0;

            ax -= (wx0 * wy0 * wz0 * (potential[iz0 + Nz * (iy0 + Ny * ix1)] - potential[iz0 + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz0 * (potential[iz0 + Nz * (iy0 + Ny * ((ix1 + 1) % Nx))] - potential[iz0 + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz0 * (potential[iz0 + Nz * (iy1 + Ny * ix1)] - potential[iz0 + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz0 * (potential[iz0 + Nz * (iy1 + Ny * ((ix1 + 1) % Nx))] - potential[iz0 + Nz * (iy1 + Ny * ix1)]) +
                   wx0 * wy0 * wz1 * (potential[iz1 + Nz * (iy0 + Ny * ix1)] - potential[iz1 + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz1 * (potential[iz1 + Nz * (iy0 + Ny * ((ix1 + 1) % Nx))] - potential[iz1 + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz1 * (potential[iz1 + Nz * (iy1 + Ny * ix1)] - potential[iz1 + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz1 * (potential[iz1 + Nz * (iy1 + Ny * ((ix1 + 1) % Nx))] - potential[iz1 + Nz * (iy1 + Ny * ix1)])) / (2 * dx);

            ay -= (wx0 * wy0 * wz0 * (potential[iz0 + Nz * (iy1 + Ny * ix0)] - potential[iz0 + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz0 * (potential[iz0 + Nz * (iy1 + Ny * ix1)] - potential[iz0 + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz0 * (potential[iz0 + Nz * (((iy1 + 1) % Ny) + Ny * ix0)] - potential[iz0 + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz0 * (potential[iz0 + Nz * (((iy1 + 1) % Ny) + Ny * ix1)] - potential[iz0 + Nz * (iy1 + Ny * ix1)]) +
                   wx0 * wy0 * wz1 * (potential[iz1 + Nz * (iy1 + Ny * ix0)] - potential[iz1 + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz1 * (potential[iz1 + Nz * (iy1 + Ny * ix1)] - potential[iz1 + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz1 * (potential[iz1 + Nz * (((iy1 + 1) % Ny) + Ny * ix0)] - potential[iz1 + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz1 * (potential[iz1 + Nz * (((iy1 + 1) % Ny) + Ny * ix1)] - potential[iz1 + Nz * (iy1 + Ny * ix1)])) / (2 * dy);

            az -= (wx0 * wy0 * wz0 * (potential[((iz1) % Nz) + Nz * (iy0 + Ny * ix0)] - potential[iz0 + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz0 * (potential[((iz1) % Nz) + Nz * (iy0 + Ny * ix1)] - potential[iz0 + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz0 * (potential[((iz1) % Nz) + Nz * (iy1 + Ny * ix0)] - potential[iz0 + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz0 * (potential[((iz1) % Nz) + Nz * (iy1 + Ny * ix1)] - potential[iz0 + Nz * (iy1 + Ny * ix1)]) +
                   wx0 * wy0 * wz1 * (potential[((iz1 + 1) % Nz) + Nz * (iy0 + Ny * ix0)] - potential[((iz1) % Nz) + Nz * (iy0 + Ny * ix0)]) +
                   wx1 * wy0 * wz1 * (potential[((iz1 + 1) % Nz) + Nz * (iy0 + Ny * ix1)] - potential[((iz1) % Nz) + Nz * (iy0 + Ny * ix1)]) +
                   wx0 * wy1 * wz1 * (potential[((iz1 + 1) % Nz) + Nz * (iy1 + Ny * ix0)] - potential[((iz1) % Nz) + Nz * (iy1 + Ny * ix0)]) +
                   wx1 * wy1 * wz1 * (potential[((iz1 + 1) % Nz) + Nz * (iy1 + Ny * ix1)] - potential[((iz1) % Nz) + Nz * (iy1 + Ny * ix1)])) / (2 * dz);

            p.vx += ax * dt;
            p.vy += ay * dt;
            p.vz += az * dt;
        } else {
            // NGP acceleration computation
            double dphi_dx = (potential[iz + Nz * (iy + Ny * ((ix + 1) % Nx))] -
                              potential[iz + Nz * (iy + Ny * ((ix - 1 + Nx) % Nx))]) / (2 * dx);
            double dphi_dy = (potential[iz + Nz * (((iy + 1) % Ny) + Ny * ix)] -
                              potential[iz + Nz * (((iy - 1 + Ny) % Ny) + Ny * ix)]) / (2 * dy);
            double dphi_dz = (potential[((iz + 1) % Nz) + Nz * (iy + Ny * ix)] -
                              potential[((iz - 1 + Nz) % Nz) + Nz * (iy + Ny * ix)]) / (2 * dz);

            p.vx -= dphi_dx * dt;
            p.vy -= dphi_dy * dt;
            p.vz -= dphi_dz * dt;
        }
    }
}

void update_positions(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;

        p.x = fmod(p.x + Lx, Lx);
        p.y = fmod(p.y + Ly, Ly);
        p.z = fmod(p.z + Lz, Lz);
    }
}

int main() {
    std::vector<Particle> particles(n_particles);
    
    for (int i = 0; i < 100; i++) particles.push_back({Lx / 4, Ly / 2, 0, 0.0, 0.0, 0.0});
    for (int i = 0; i < 100; i++) particles.push_back({3 * Lx / 4, Ly / 2, 0, 0.0, 0.0, 0.0});
    std::vector<double> density(Nx * Ny * Nz, 0.0);
    std::vector<double> potential(Nx * Ny * Nz, 0.0);

    int steps = 500;
    bool useCIC = true;

    for (int step = 0; step < steps; ++step) {
        map_to_grid(particles, density, useCIC);
        solve_poisson_3d(density, potential);
        compute_acceleration(potential, particles, useCIC);
        update_positions(particles);

        if (step % 10 == 0) {
            // Save density slice at z=0
            std::ofstream density_out("density_z0_step_" + std::to_string(step) + ".dat");
            for (int iy = 0; iy < Ny; ++iy) {
                for (int ix = 0; ix < Nx; ++ix) {
                    int idx = 0 + Nz * (iy + Ny * ix); // z=0 slice
                    density_out << density[idx] << " ";
                }
                density_out << "\n";
            }
            density_out.close();

            // Save potential slice at z=0
            std::ofstream potential_out("potential_z0_step_" + std::to_string(step) + ".dat");
            for (int iy = 0; iy < Ny; ++iy) {
                for (int ix = 0; ix < Nx; ++ix) {
                    int idx = 0 + Nz * (iy + Ny * ix); // z=0 slice
                    potential_out << potential[idx] << " ";
                }
                potential_out << "\n";
            }
            potential_out.close();

            std::cout << "Step " << step << ": Data saved for density and potential slices.\n";
        }

    }

    return 0;
}
