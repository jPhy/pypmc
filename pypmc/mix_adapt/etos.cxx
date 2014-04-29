/* This code is based on the EToS spike sorting system.
 * Copyright (C) 2008-2010 Takashi Takekawa (RIKEN)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */

#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// TODO not portable!
#define F77_FUNC(name,NAME) name ## _

#define USE_ROBUST 1

#define USE_FIND_ROOT 1

extern "C" {

#define DPOTRF_F77 F77_FUNC(dpotrf, DPOTRF)
   void DPOTRF_F77(char const& uplo, int const& n,
                   double* a, int const& lda, int& info);
#define DPOTRI_F77 F77_FUNC(dpotri, DPOTRI)
   void DPOTRI_F77(char const& uplo, int const& n,
                   double* a, int const& lda, int& info);
#define DTRSV_F77 F77_FUNC(dtrsv, DTRSV)
   void DTRSV_F77(char const& uplo, char const& trans, char const& diag,
                  int const& n, double const* a, int const& lda,
                  double* x, int const& incx);
#define DSYEV_F77 F77_FUNC(dsyev, DSYEV)
   void DSYEV_F77(char const& jobz, char const& uplo, int const& n,
                  double* a, int const& lda, double* w,
                  double* work, int const& lwork, int& info);

}

namespace etos
{
   /*
    * mathutils
    */

   double const PI(std::acos(-1.0));

   template <typename F>
   double
   find_root(F const& f, double x0, double x1)
   {
      double f0(f(x0));
      double f1(f(x1));
      if (f0 * f1 > 0.0) {
         if (std::abs(f0) < std::abs(f1)) {
            return x0;
         } else {
            return x1;
         }
      }
      while (f0 != 0.0) {
         double const dx(0.1 * (x1 - x0));
         double const fs(f(x0 + dx));
         double const
            xn(((f0 * (fs - f0) >= 0.0) || (f0 * (fs - 0.9 * f0) >= 0.0))
               ?  0.5 * (x0 + x1) : x0 - dx * f0 / (fs - f0));
         if (x0 == xn) {
            break;
         }
         double const fn(f(xn));
         if (f0 * fn < 0.0) {
            x1 = x0;
         }
         x0 = xn;
         f0 = fn;
      }
      return x0;
   }

   double
   digamma(double x)
   {
      double const D02((    1.0 /    6.0) /  -2.0);
      double const D04((   -1.0 /   30.0) /  -4.0);
      double const D06((    1.0 /   42.0) /  -6.0);
      double const D08((   -1.0 /   30.0) /  -8.0);
      double const D10((    5.0 /   66.0) / -10.0);
      double const D12(( -691.0 / 2730.0) / -12.0);
      double const D14((    7.0 /    6.0) / -14.0);
      double const D16((-3617.0 /  510.0) / -16.0);

      double v(0.0);
      while (x < 8.0) {
         v += 1.0 / x;
         x += 1.0;
      }
      double const x2(x * x);
      return ((((((((D16 / x2 + D14) / x2 + D12) / x2 + D10) / x2 + D08) / x2
                 + D06) / x2 + D04) / x2 + D02) / x2 - 0.5 / x + std::log(x)) - v;
   }

   /*
    * emprivate
    */

   namespace detail
   {

      struct kappa_comp
      {
         template <typename T>
         bool operator()(T const* x, T const* y) const
         {
            return x->kappa_ < y->kappa_;
         }
      };

   }

   template <typename T>
   double
   calc_z_and_score(std::vector<T*>& clusters_)
   {
      std::sort(clusters_.rbegin(), clusters_.rend(), detail::kappa_comp());
      int const n_clusters(clusters_.size());
      int const n_data(clusters_[0]->z_.size());
      double score(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:score)
#endif
      for (int n = 0; n < n_data; ++n) {
         double max_p(clusters_[0]->p_[n]);
         for (int k(1); k < n_clusters; ++k) {
            if (clusters_[k]->p_[n] > max_p) {
               max_p = clusters_[k]->p_[n];
            }
         }
         double sum_z(0.0);
         for (int k(0); k < n_clusters; ++k) {
            clusters_[k]->z_[n] = std::exp(clusters_[k]->p_[n] - max_p);
            sum_z += clusters_[k]->z_[n];
         }
         for (int k(0); k < n_clusters; ++k) {
            clusters_[k]->z_[n] = clusters_[k]->z_[n] / sum_z;
         }
         score += max_p + std::log(sum_z);
      }
      if (n_clusters == 1) {
         clusters_[0]->score_ = 0.0;
      } else {
         for (int k(0); k < n_clusters; ++k) {
            double scorek(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:scorek)
#endif
            for (int n = 0; n < n_data; ++n) {
               scorek += -std::log(1.0 - clusters_[k]->z_[n]);
            }
            clusters_[k]->score_ += scorek;
         }
      }
      return score;
   }

   /**
    * Perform m step for components individually and
    * remove a component if its Cholesky decomposition fails.
    * @param c
    * @param x
    * @param p
    * @param beta
    */
   template <typename T, typename X, typename P>
   void
   exec_m_step(std::vector<T*>& c, X const& x, P const& p, double beta)
   {
      for (std::size_t k(0); k < c.size();) {
         if (c[k]->m_step(x, p, beta)) {
            ++k;
         } else {
            std::swap(c[k], c.back());
            c.resize(c.size() - 1);
         }
      }
   }

   struct nu_equation
   {
      double const c_;
      nu_equation(double xi) : c_(1.0 - xi) {}
      double operator()(double x) const
      {
         double const xh(0.5 * x);
         return std::log(xh) - digamma(xh) + c_;
      }
   };

   inline double
   calc_nu_peak(double xi)
   {
#if USE_FIND_ROOT
      return find_root(nu_equation(xi), 1e-3, 1e3);
#else
#include "robust_table1.h"
      int const max_idx(sizeof(TABLE) / sizeof(TABLE[0]) - 1);
      double const idx(XIA * std::log(XIB * (xi - XI0) + 1.0));
      int const idx_int(static_cast<int>(std::floor(idx)));
      double nu_peak;
      if (idx < 0) {
         nu_peak = find_root(nu_equation(xi), TABLE[0], 1e10);
      } else if (max_idx <= idx) {
         nu_peak = find_root(nu_equation(xi), 1e-10, TABLE[max_idx]);
      } else {
         double const rate(idx - idx_int);
         nu_peak = (1.0 - rate) * TABLE[idx_int] + rate * TABLE[idx_int + 1];
      }
      return nu_peak;
#endif
   }

   inline void
   calc_nu_functions(double xi, double& nu_const, double& nu_bar, double& nu_hat)
   {
#include "robust_table2.h"
      int const max_idx(sizeof(TABLE) / sizeof(TABLE[0]) - 1);
      double const idx(XIA * std::log(XIB * (xi - XI0) + 1.0));
      int const idx_int(static_cast<int>(std::floor(idx)));
      if (idx < 0) {
         std::clog << "# small xi: " << xi << '\n';
         nu_const = TABLE[0][0];
         nu_bar = TABLE[0][1];
         nu_hat = TABLE[0][2];
      } else if (max_idx <= idx) {
         std::clog << "# large xi: " << xi << '\n';
         nu_const = TABLE[max_idx][0];
         nu_bar = TABLE[max_idx][1];
         nu_hat = TABLE[max_idx][2];
      } else {
         double const rate(idx - idx_int);
         nu_const = (1.0 - rate) * TABLE[idx_int][0]
            + rate * TABLE[idx_int + 1][0];
         nu_bar = (1.0 - rate) * TABLE[idx_int][1]
            + rate * TABLE[idx_int + 1][1];
         nu_hat = (1.0 - rate) * TABLE[idx_int][2]
            + rate * TABLE[idx_int + 1][2];
      }
   }


   /*
    * lmatrix
    */
   template <typename T>
   class lmatrix
   {
   public:
      explicit lmatrix(int size = 0);

      void resize(int size);

      int size() const;

      T* operator[](int i);
      T const* operator[](int i) const;

   private:
      std::vector<T> x_;
      int size_;
   };

   int eigenvalue(lmatrix<double>& lmat, double* ev);

   int cholesky(lmatrix<double>& mat);
   int cholesky(lmatrix<double> const& smat, lmatrix<double>& lmat);

   int inverse(lmatrix<double>& mat);
   int inverse(lmatrix<double> const& lmat, lmatrix<double>& imat);

   double mahalanobis(lmatrix<double> const& lmat, double* x);

   void pca(double* x, int num, int dim, double* cr, bool centering = true);

   void normalize(double* x, int num, int dim);

   template <typename T>
   lmatrix<T>::lmatrix(int size) :
      x_(size * size), size_(size)
   {
   }

   template <typename T>
   void
   lmatrix<T>::resize(int size)
   {
      if (size_ != size) {
         x_.resize(size * size);
         size_ = size;
      }
   }

   template <typename T>
   inline int
   lmatrix<T>::size() const
   {
      return size_;
   }

   template <typename T>
   inline T*
   lmatrix<T>::operator[](int i)
   {
      return &x_[i * size_];
   }

   template <typename T>
   inline T const*
   lmatrix<T>::operator[](int i) const
   {
      return &x_[i * size_];
   }

   double
   mahalanobis(lmatrix<double> const& lmat, double* x)
   {
      DTRSV_F77('U', 'T', 'N', lmat.size(), &lmat[0][0], lmat.size(), x, 1);
      double sum2(0.0);
      for (int i(0); i < lmat.size(); ++i) {
         sum2 += x[i] * x[i];
      }
      return sum2;
   }

   int
   cholesky(lmatrix<double>& mat)
   {
      int info;
      DPOTRF_F77('U', mat.size(), &mat[0][0], mat.size(), info);
      return info;
   }

   int
   cholesky(lmatrix<double> const& smat, lmatrix<double>& lmat)
   {
      lmat = smat;
      return cholesky(lmat);
   }

   int
   inverse(lmatrix<double>& mat)
   {
      int info;
      DPOTRI_F77('U', mat.size(), &mat[0][0], mat.size(), info);
      return info;
   }

   int
   inverse(lmatrix<double> const& lmat, lmatrix<double>& imat)
   {
      imat = lmat;
      return inverse(imat);
   }

   /**
    * embase
    */
   class embase
   {
   public:
      embase(double const* x, int num, int dim);

      int n_data() const;
      int dim() const;
      double const* operator[](int n) const;

   protected:
      /// data matrix num_ x dim_
      double const* x_;
      int num_;
      int dim_;
   };

   inline int
   embase::n_data() const
   {
      return num_;
   }

   inline int
   embase::dim() const
   {
      return dim_;
   }

   inline double const*
   embase::operator[](int n) const
   {
      return &x_[n * dim_];
   }

   embase::embase(double const* x, int num, int dim) :
      x_(x), num_(num), dim_(dim)
   {
   }

   /**
    * robust_vb
    * todo prior for individual components including mu
    * todo removing comp. with negative contribution to F:
    *      - are samples redistributed to other components?
    *      - why is negative bad?
    *      - why only smallest removed?
    * todo
    */
   class robust_vb : public embase
   {
   public:
      /**
       * Parameters of the variational Bayes approximating
       * distribution for a single component.
       */
      struct param_t
      {
         /// Dirichlet
         double kappa_;

         /// inverse Wishart dof parameter
         double gamma_;

         /// precision-scale parameter in the normal distribution
         double eta_;

         /// exponential scale
         double xi_;

         /// normal mean
         std::vector<double> mu_;

         /// normal covariance
         lmatrix<double> sigma_;
      };

      typedef param_t prior_t;

      struct cluster_t : param_t
      {
         /// bidirection cluster identification
         int id_;

         /// used in E step, what does it mean??
         double p0_;

         /// latent variables
         std::vector<double> p_;
         std::vector<double> u_;
         std::vector<double> v_;
         std::vector<double> z_;

         /// Cholesky decomp
         lmatrix<double> l_;

         /// inverse
         lmatrix<double> inv_;

         double score_;

         double d(double const* x, double* work) const;

         double e_step(embase const& x,
                       prior_t const& prior, double p0, double beta,
                       double sum_lgamma0, double logdet0);

         bool m_step(embase const& x, prior_t const& prior, double beta);
      };

      /**
       * Ctor.
       *
       * @param x Samples in num x dim matrix.
       * @param num Number of samples.
       * @param dim Dimension of each sample.
       */
      robust_vb(double const* x, int num, int dim);

      /**
       * Set prior values, identical for all components.
       *
       * @param kappa Dirichlet
       * @param gamma Wishard dof
       * @param eta scale of precision matrix in normal
       * @param nu student's t dof
       * @param sigma std. dev on diagonal. Use 1.0 for unit matrix.
       */
      void set_param(double kappa, double gamma, double eta,
                     double nu, double sigma);

      // void load_param(std::istream& stream);
      // void save_param(std::ostream& stream) const;

      // void load_model(std::istream& stream);
      // void save_model(std::ostream& stream) const;

      /**
       * Load student's t mixture density
       *
       * @param n_clusters
       * @param id
       * @param kappa
       * @param gamma
       * @param eta
       * @param xi
       * @param mu
       * @param sigma covariance matrix stored as lower triangular matrix.
       *              cov[0,0] at sigma[0], cov[1,0] is at sigma[1], cov[1,1] at sigma[2] ...
       */
       void load_model(int n_clusters, std::vector<int> id,
                       std::vector<double> kappa, std::vector<double> gamma,
                       std::vector<double> eta, std::vector<double> xi,
                       std::vector<std::vector<double>> mu, std::vector<std::vector<double>> sigma);
       void save_model(std::ostream& stream) const;

      void remove_cluster(int k);

      // only used for multimodality
//      void split_cluster(int k);

      double e_step(double beta = 1.0);

      void m_step(double beta = 1.0);

      prior_t const& prior() const;
      int n_clusters() const;
      cluster_t const& cluster(int k) const;

      int id(int k) const;
      double p(int n, int k) const;
      double z(int n, int k) const;

   private:
      // identical prior for every component
      prior_t prior_;

      double sum_lgamma0_;
      double logdet0_;

      // just a data holder?
      std::vector<cluster_t> cdata_;

      // all parameters defining a cluster
      std::vector<cluster_t*> clusters_;

      double e_step_(double beta = 1.0);
   };

   inline robust_vb::prior_t const&
   robust_vb::prior() const
   {
      return prior_;
   }

   inline int
   robust_vb::n_clusters() const
   {
      return clusters_.size();
   }

   inline robust_vb::cluster_t const&
   robust_vb::cluster(int k) const
   {
      return *clusters_[k];
   }

   inline int
   robust_vb::id(int k) const
   {
      return clusters_[k]->id_;
   }

   inline double
   robust_vb::p(int n, int k) const
   {
      return std::exp(clusters_[k]->p_[n] - clusters_[k]->p0_);
   }

   inline double
   robust_vb::z(int n, int k) const
   {
      return clusters_[k]->z_[n];
   }

   robust_vb::robust_vb(double const* x, int num, int dim) :
      embase(x, num, dim)
   {
      prior_.mu_.resize(dim);
      prior_.sigma_.resize(dim);
   }

   void
   robust_vb::set_param(double kappa, double gamma, double eta,
                        double nu, double sigma)
   {
      sigma = sigma * sigma;
      prior_.kappa_ = kappa;
      prior_.gamma_ = (gamma < 0.0 ? -gamma * n_data() : gamma);
      if (prior_.gamma_ < dim()) {
         prior_.gamma_ = dim();
      }
      prior_.eta_ = eta;
      prior_.xi_ = (nu == 0.0) ? 0.0 : (1.0 / nu);
      for (int i(0); i < dim(); ++i) {
         prior_.mu_[i] = 0.0;
      }
      for (int i(0); i < dim(); ++i) {
         for (int j(0); j < i; ++j) {
            prior_.sigma_[i][j] = 0.0;
         }
         prior_.sigma_[i][i] = sigma;
      }
      sum_lgamma0_ = 0.0;
      for (int i(0); i < dim(); ++i) {
         double const g0i(prior_.gamma_ - i);
         sum_lgamma0_ += lgamma(g0i);
      }
      logdet0_ = dim() * std::log(prior_.gamma_ * sigma);
   }

   void
   robust_vb::load_model(int n_clusters, std::vector<int> id,
         std::vector<double> kappa, std::vector<double> gamma,
         std::vector<double> eta, std::vector<double> xi,
         std::vector<std::vector<double>> mu, std::vector<std::vector<double>> sigma)
   {
      // todo check dim, vector lengths
      if (n_clusters > int(cdata_.size())) {
         cdata_.resize(n_clusters);
      }
      clusters_.resize(n_clusters);
      for (int k(0); k < n_clusters; ++k) {
         clusters_[k] = &cdata_[k];
         cluster_t& c(*clusters_[k]);
         c.id_ = id[k]; c.kappa_ = kappa[k]; c.gamma_ = gamma[k]; c.eta_ = eta[k], c.xi_ = xi[k];
         c.mu_.resize(dim());
         // todo size check
         std::copy(mu[k].begin(), mu[k].end(), c.mu_.begin());
         c.sigma_.resize(dim());
         // todo size check
         for (int i(0); i < dim(); ++i) {
           for (int j(0); j <= i; ++j) {
              // lower diagonal matrix stored such that
              // first element in row at sum of previous elements
              // + offset: i(i+1)/2 + j
              c.sigma_[i][j] = sigma[k][i * (i + 1) / 2 + j];
           }
         }

         c.p_.resize(n_data());
         c.u_.resize(n_data());
         c.v_.resize(n_data());
         c.z_.resize(n_data());
         c.l_.resize(dim());
         c.inv_.resize(dim());
         if (cholesky(c.sigma_, c.l_) != 0) {
            std::ostringstream stream;
            stream << "invalid load: " << c.id_ << '\n';
            throw std::runtime_error(stream.str());
         }
      }
   }

   void
   robust_vb::save_model(std::ostream& stream) const
   {
     stream << n_clusters() << " components\n";
     for (int k(0); k < n_clusters(); ++k) {
       cluster_t const& c(*clusters_[k]);
       stream << "component " << c.id_ << '\n';
       const double nu = calc_nu_peak(c.xi_);
       stream << "kappa\tgamma\teta\tnu\n";
       stream << c.kappa_ << ' '
         << c.gamma_ << ' ' << c.eta_ << ' ' << nu << '\n';
       stream << "mu\n";
       for (int i(0); i < dim(); ++i) {
         stream << c.mu_[i] << (i + 1 == dim() ? '\n' : ' ');
       }
       stream << "sigma\n";
       for (int i(0); i < dim(); ++i) {
          for (int j(0); j <= i; ++j) {
             // rescale for parameters at mode
             stream << c.sigma_[i][j] << (j == i ? '\n' : ' ');

         }
       }
       stream << "weights\n";
       // compute covariance at mode
//       lmatrix<double> cov(c.sigma_);

       // inverse expects Cholesky factor, so only need
       // lower triangular matrix. Since we rescale to get the mode
       // the Cholesky root is multiplied by the sqrt(...)
//       const double scale = std::sqrt(c.gamma_ - dim());
//       for (int i(0); i < dim(); ++i) {
//         for (int j(0); j <= i; ++j) {
//            cov[i][j] *= scale;
//         }
//       }
//       inverse(cov);
//       stream << "cov\n";
//       for (int i(0); i < dim(); ++i) {
//         for (int j(0); j <= i; ++j) {
//           stream << cov[i][j] << (j == i ? '\n' : ' ');
//         }
//       }
     }
   }

   void
   robust_vb::remove_cluster(int k)
   {
      std::swap(clusters_[k], clusters_.back());
      clusters_.resize(clusters_.size() - 1);
   }

   double
   robust_vb::e_step_(double beta)
   {
      double penalty;
      double score;
      double sum_kappa(0.0);
      for (int k(0); k < n_clusters(); ++k) {
         sum_kappa += clusters_[k]->kappa_;
      }
      double const kappa0_dash(beta * prior_.kappa_ - beta + 1.0);
      double const digamma_sum_kappa(digamma(sum_kappa));
      double const penalty0(
                            + lgamma(sum_kappa)
                            - beta * lgamma(n_clusters() * prior_.kappa_)
                            - (sum_kappa - n_clusters() * kappa0_dash) * digamma_sum_kappa
                            );
      double const p0(-0.5 * dim() * std::log(PI) - digamma_sum_kappa);
      penalty = penalty0;
      for (int k(0); k < n_clusters(); ++k) {
         double const penaltyk(clusters_[k]->e_step(*this, prior_, p0, beta,
                                                    sum_lgamma0_, logdet0_));
         penalty += penaltyk;
         double const sum_kappak(sum_kappa - clusters_[k]->kappa_);
         double const penalty0_dash(
                                    + lgamma(sum_kappak)
                                    - beta * lgamma((n_clusters() - 1) * prior_.kappa_)
                                    - (sum_kappak - (n_clusters() - 1) * kappa0_dash) * digamma(sum_kappak)
                                    );
         clusters_[k]->score_ = - penaltyk - (penalty0 - penalty0_dash);
      }
      score = calc_z_and_score(clusters_);
      return score - penalty;
   }

   double
   robust_vb::e_step(double beta)
   {
      double score(e_step_(beta));
//      double min_score(0.0);
//      int min_k(-1);
//      for (int k(0); k < n_clusters(); ++k) {
//         if (clusters_[k]->score_ < min_score) {
//            min_score = clusters_[k]->score_;
//            min_k = k;
//         }
//      }
//      if (min_k != -1) {
//         remove_cluster(min_k);
//         score = e_step_(beta);
//      }
      return score;
   }

   void
   robust_vb::m_step(double beta)
   {
      exec_m_step(clusters_, *this, prior_, beta);
   }

   double
   robust_vb::cluster_t::d(double const* x, double* work) const
   {
      for (int i(0); i < sigma_.size(); ++i) {
         work[i] = x[i] - mu_[i];
      }
      return mahalanobis(l_, &work[0]);
   }

   double
   robust_vb::cluster_t::e_step(embase const& x,
                                prior_t const& prior, double p0, double beta,
                                double sum_lgamma0, double logdet0)
   {
      int const num(x.n_data());
      int const dim(x.dim());

      inverse(l_, inv_);

      double sum_lgamma(0.0);
      double sum_digamma(0.0);
      double tr(0.0);
      double det(1.0);
      for (int i(0); i < dim; ++i) {
         double const gi(0.5 * (gamma_ - i));
         sum_lgamma += lgamma(gi);
         sum_digamma += digamma(gi);
         for (int j(0); j < i; ++j) {
            tr += 2.0 * inv_[i][j] * prior.sigma_[i][j];
         }
         tr += inv_[i][i] * prior.sigma_[i][i];
         det *= l_[i][i];
      }
      double const digamma_kappa(digamma(kappa_));
      double const logdet(dim * std::log(gamma_) + 2.0 * std::log(det));

      double penalty(0.0);
      if (prior.xi_ == 0.0) {
         double const nu_peak(calc_nu_peak(xi_));

         double a(0.5 * (nu_peak + dim));
         a = beta * a + (1.0 - beta); // for annealing
         double const digamma_a(digamma(a));

         double const nuh(0.5 * nu_peak);
         double c(
                  + p0
                  + digamma_kappa
                  - 0.5 * logdet
                  + 0.5 * sum_digamma
                  + nuh * std::log(nuh)
                  - lgamma(nuh)
                  );
         c *= beta; // for annealing
         c += lgamma(a);

         double const cb(nu_peak + dim / eta_);
         p0_ = c - a * std::log(0.5 * cb);
#ifdef _OPENMP
#pragma omp parallel
#endif
         {
            std::vector<double> work(dim);
#ifdef _OPENMP
#pragma omp for
#endif
            for (int n = 0; n < num; ++n) {
               double const md(d(&x[n][0], &work[0]));
               double b(0.5 * (cb + md));
               b *= beta; // for annealing
               double const log_b(std::log(b));
               p_[n] = c - a * log_b;
               u_[n] = a / b;
               v_[n] = digamma_a - log_b;
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
               double const kappa0(beta * prior.kappa_ - beta + 1.0);
               double const penalty_alpha(
                                          - lgamma(kappa_) + beta * lgamma(kappa0)
                                          + (kappa_ - kappa0) * digamma_kappa
                                          );
               double const gamma0(beta * prior.gamma_ + dim - beta * dim);
               double const penalty_mu_sigma(
                                             - (1.0 - beta) * (dim * dim + dim) * std::log(4.0 * PI)
                                             + 0.5 * dim * (std::log(eta_) - beta * std::log(prior.eta_))
                                             - 0.5 * dim * (eta_ - beta * prior.eta_) / eta_
                                             + 0.5 * beta * prior.eta_ * d(&prior.mu_[0], &work[0])
                                             - sum_lgamma + beta * sum_lgamma0
                                             + 0.5 * (gamma0 * logdet - beta * prior.gamma_ * logdet0)
                                             + 0.5 * (gamma_ - gamma0) * sum_digamma
                                             - 0.5 * (gamma_ * dim - beta * prior.gamma_ * tr)
                                             );
               penalty = (penalty_alpha + penalty_mu_sigma);
            }
         }
      } else {
         double nu_const;
         double nu_bar;
         double nu_hat;
         calc_nu_functions(xi_, nu_const, nu_bar, nu_hat);

         double a(0.5 * (nu_bar + dim));
         a = beta * a + (1.0 - beta); // for annealing
         double const digamma_a(digamma(a));

         double c(p0 + digamma_kappa - 0.5 * logdet + 0.5 * sum_digamma + nu_hat);
         c *= beta; // for annealing
         c += lgamma(a);

         double const cb(nu_bar + dim / eta_);
         p0_ = c - a * std::log(0.5 * cb);
#ifdef _OPENMP
#pragma omp parallel
#endif
         {
            std::vector<double> work(dim);
#ifdef _OPENMP
#pragma omp for
#endif
            for (int n = 0; n < num; ++n) {
               double const md(d(&x[n][0], &work[0]));
               double b(0.5 * (cb + md));
               b *= beta; // for annealing
               double const log_b(std::log(b));
               p_[n] = c - a * log_b;
               u_[n] = a / b;
               v_[n] = digamma_a - log_b;
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
               double const kappa0(beta * prior.kappa_ - beta + 1.0);
               double const penalty_alpha(
                                          - lgamma(kappa_) + beta * lgamma(kappa0)
                                          + (kappa_ - kappa0) * digamma_kappa
                                          );
               double const gamma0(beta * prior.gamma_ + dim - beta * dim);
               double const penalty_mu_sigma(
                                             - (1.0 - beta) * (dim * dim + dim) * std::log(4.0 * PI)
                                             + 0.5 * dim * (std::log(eta_) - beta * std::log(prior.eta_))
                                             - 0.5 * dim * (eta_ - beta * prior.eta_) / eta_
                                             + 0.5 * beta * prior.eta_ * d(&prior.mu_[0], &work[0])
                                             - sum_lgamma + beta * sum_lgamma0
                                             + 0.5 * (gamma0 * logdet - beta * prior.gamma_ * logdet0)
                                             + 0.5 * (gamma_ - gamma0) * sum_digamma
                                             - 0.5 * (gamma_ * dim - beta * prior.gamma_ * tr)
                                             );
               double const penalty_nu(
                                       + nu_hat
                                       - (xi_ - beta * prior.xi_) * nu_bar
                                       - std::log(nu_const)
                                       - beta * std::log(prior.xi_)
                                       );
               penalty = (penalty_alpha + penalty_mu_sigma + penalty_nu);
            }
         }
      }
      return penalty;
   }

   bool
   robust_vb::cluster_t::m_step(embase const& x, prior_t const& prior, double beta)
   {
      int const num(x.n_data());
      int const dim(x.dim());

      double sumz(0.0);
      double sumu(0.0);
      double sumv(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumz) reduction(+:sumu) reduction(+:sumv)
#endif
      for (int n = 0; n < num; ++n) {
         sumz += z_[n];
         u_[n] *= z_[n];
         sumu += u_[n];
         sumv += z_[n] * v_[n];
      }
      if ((sumz < 1.0) || (sumu < 1e-6)) {
         return false;
      }

      kappa_ = prior.kappa_ + sumz;
      gamma_ = prior.gamma_ + sumz;
      eta_ = prior.eta_ + sumu;
      xi_ = prior.xi_ + 0.5 * (sumu - sumv) / sumz;

      std::vector<double> mubar(dim);
      std::vector<double> dmubar(dim);
      for (int i(0); i < dim; ++i) {
         double sum1(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum1)
#endif
         for (int n = 0; n < num; ++n) {
            sum1 += u_[n] * x[n][i];
         }
         mu_[i] = (prior.eta_ * prior.mu_[i] + sum1) / eta_;
         mubar[i] = sum1 / sumu;
         dmubar[i] = mubar[i] - prior.mu_[i];
      }
      double const ck(prior.eta_ * sumu / eta_);

      // annealing
      kappa_ = beta * kappa_ + (1.0 - beta);
      gamma_ = beta * gamma_ + (1.0 - beta) * (dim + 1);
      eta_ = beta * eta_;
      //xi_ = beta * xi_;
      for (int i(0); i < dim; ++i) {
         for (int j(0); j <= i; ++j) {
            double sum2(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum2)
#endif
            for (int n = 0; n < num; ++n) {
               sum2 += u_[n] * x[n][i] * x[n][j];
            }
            sigma_[i][j] =
               (prior.gamma_ * prior.sigma_[i][j]
                + (sum2 - sumu * mubar[i] * mubar[j])
                + ck * dmubar[i] * dmubar[j]) / gamma_;
            // anneanling
            sigma_[i][j] = beta * sigma_[i][j];
         }
      }
      return (cholesky(sigma_, l_) == 0);
   }

   template <typename T>
   double
   classify(T& alg, double eps, double temp0, double dtemp,
            std::ostream& log, bool remove)
   {
      if (temp0 < 1.0) {
         throw std::runtime_error("invalid temp0 value");
      }
      if ((dtemp < 0.0) || (1.0 < dtemp)) {
         throw std::runtime_error("invalid temp setting");
      }
      // alg.save_param(log);
      log << "START " << alg.n_clusters() << '\n';
      int n(0);
      double score(-std::numeric_limits<double>::max());

      /* annealing */
      for (double temp(temp0); temp > 1.0; ++n) {
         double const beta(1.0 / temp);
         double const newscore(alg.e_step(beta));
         double const diff((newscore - score) / alg.n_data());
         score = newscore;
         alg.m_step(beta);
         if ((n % 50 == 0) && (n != 0)) {
            log << '\n' << temp << '\t'
                << alg.n_clusters() << '\t' << score << '\t' << diff << '\n';
         }
         if (diff < 0.0) {
            log << '!' << std::flush;
         } else if (diff <= 10.0 * eps) {
            temp *= dtemp;
            log << '-' << std::flush;
         } else {
            log << '/' << std::flush;
         }
         if (alg.n_clusters() == 0) {
            throw std::runtime_error("cluster number is zero");
         }
      }

      /* regular EM steps */
      for (;; ++n) {
         double const newscore(alg.e_step());
         log << "after E step: K=" << alg.n_clusters() << '\n';
         double const diff((newscore - score) / alg.n_data());
         score = newscore;
         if ((0.0 <= diff) && (diff <= eps)) {
            log << "\nRESULT "
                << alg.n_clusters() << '\t' << score << '\t' << diff << '\n';
            break;
         }
         alg.m_step();
         log << "afte M step K= " << alg.n_clusters() << '\n';
         if ((n % 1 == 0) && (n != 0)) {
            log << '\n'
                << alg.n_clusters() << '\t' << score << '\t' << diff << '\t' << alg.cluster(0).kappa_ << '\n';
         }
         if (diff < 0.0) {
            log << '!' << std::flush;
         } else {
            log << '*' << std::flush;
         }
         if (alg.n_clusters() == 0) {
            throw std::runtime_error("cluster number is zero");
         }
         // if (savemodelfile != 0) {
         //    std::ofstream modelstream(savemodelfile);
         //    alg.save_param(modelstream);
         //    alg.save_model(modelstream);
         // }
      }
      if (remove) {
         for (;;) {
            score = alg.e_step();
            double current_score(std::numeric_limits<double>::max());
            alg.remove_cluster(alg.n_clusters() - 1);
            log << "REMOVE SMALLEST CLUSTER " << alg.n_clusters() << '\n';
            for (int n(0);; ++n) {
               double const newscore(alg.e_step());
               double const diff((newscore - current_score) / alg.n_data());
               current_score = newscore;
               if ((0.0 <= diff) && (diff <= eps)) {
                  log << "\nRESULT "
                      << alg.n_clusters() << '\t' << newscore << '\t' << diff << '\n';
                  // todo stop after the best solution found, return that one instead
                  break;
               }
               alg.m_step();
               if ((n % 50 == 0) && (n != 0)) {
                  log << '\n'
                      << alg.n_clusters() << '\t' << newscore << '\t' << diff << '\n';
               }
               if (diff < 0.0) {
                  log << '!' << std::flush;
               } else {
                  log << '*' << std::flush;
               }
               if (alg.n_clusters() == 0) {
                  throw std::runtime_error("cluster number is zero");
               }
            }
            if (score > current_score) {
               break;
            }
            // if (savemodelfile != 0) {
            //    std::ofstream modelstream(savemodelfile);
            //    alg.save_param(modelstream);
            //    alg.save_model(modelstream);
            // }
         }
      }
      return score;
   }

}

int main()
{
   static const size_t N = 9;
   static const size_t D = 2;

   // define samples
   std::vector<double> x{
       -2, 3,
       2, 5,
       -1, 7,
       0, 4,
       1, 6,
      // second comp
       2, -3,
       -1, -6,
       1, -4,
       -2, -7
   };

   etos::robust_vb alg(&x[0], N, D);

   // set the prior
   // alg.load_param();
   {
       double kappa = 1e-5;
       double gamma = 3;
       double eta = 1e-5;
       double nu = 1000;
       double sigma = 1;
       alg.set_param(kappa, gamma, eta, nu, sigma);
   }

   // alg.load_model();
   static const int K = 2;
   std::vector<int> id = {0, 1};
   std::vector<double> kappa(K, 1e-5);
   std::vector<double> gamma(K, 3);
   std::vector<double> eta(K, 1e-5);
   // todo segfault if xi < 0.5
   std::vector<double> xi(K, 0.5);
   auto mu = std::vector<std::vector<double> >{
      std::vector<double>{-2, 3},
      std::vector<double>{2, -3}
   };
   // unit matrices
   auto sigma = std::vector<std::vector<double> >{
      std::vector<double>{1, 0, 1},
      std::vector<double>{1, 0, 1}
   };

   // set the starting value of posterior parameters
   alg.load_model(K, id, kappa, gamma, eta, xi, mu, sigma);

   // only one annealing step?
   double temp0 = 1;
   double dtemp = 1;

   double eps = 1e-4;
   bool remove = false;

   double score = classify(alg, eps, temp0, dtemp, std::clog, remove);
   alg.save_model(std::clog);
   return 0;
}


// Local Variables:
// compile-command: "g++ -std=c++11 -Wall -pedantic -fopenmp -g -O0 etos.cxx -llapack -lblas -o etos && ./etos"
// End:
