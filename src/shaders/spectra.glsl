// Copyright (c) 2019, 2025, Christoph Peters
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and / or other materials provided with the distribution.
//     * Neither the name of the Karlsruhe Institute of Technology nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#if __VERSION__ < 400
//! Substitute for fused multiplication and addition (not available in WebGL)
float fma(float a, float b, float c) {
    return a * b + c;
}
vec2 fma(vec2 a, vec2 b, vec2 c) {
    return a * b + c;
}
#endif


//! Applies complex conjugation to the given complex number (i.e. it flips the
//! sign of the imaginary part).
vec2 cconj(vec2 z) {
	return vec2(z.x,-z.y);
}


//! This function implements complex multiplication.
vec2 cmul(vec2 lhs, vec2 rhs) {
	return vec2(fma(lhs.x, rhs.x,-lhs.y * rhs.y), fma(lhs.x, rhs.y, lhs.y * rhs.x));
}


//! This function implements complex multiplication followed by addition.
vec2 cfma(vec2 a, vec2 b, vec2 c) {
	return vec2(fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y)));
}


//! This function computes the squared magnitude of the given complex number.
float cabs_sq(vec2 z) {
	return dot(z, z);
}


//! Implements Eq. 6 and 7 for three real trigonometric moments:
//! https://doi.org/10.1145/3306346.3322964
void trig_to_exp_moments_real_3(out vec2 out_exp_moments[3], vec3 trig_moments) {
	float moment_0_phase = fma(3.14159265, trig_moments[0],-1.57079633);
	out_exp_moments[0].y = sin(moment_0_phase);
	out_exp_moments[0].x = cos(moment_0_phase);
	out_exp_moments[0] = 0.0795774715 * out_exp_moments[0];
	out_exp_moments[1] = (trig_moments[1] * 6.28318531) * vec2(-out_exp_moments[0].y, out_exp_moments[0].x);
	out_exp_moments[2] = fma(vec2(trig_moments[2] * 6.28318531), vec2(-out_exp_moments[0].y, out_exp_moments[0].x),(trig_moments[1] * 3.14159265) * vec2(-out_exp_moments[1].y, out_exp_moments[1].x));
	out_exp_moments[0] = 2.0 * out_exp_moments[0];
}


//! Implements Levinson's algorithm with biasing for complex 3x3 Toeplitz
//! matrices (Alg. 2): https://doi.org/10.2312/mam.20191304
void levinson_3_biased(out vec2 out_solution[3], inout vec2 first_column[3]) {
	float one_minus_bias = 0.9999;
	float corrected_factor = 1.0 / (1.0 - one_minus_bias * one_minus_bias);
	out_solution[0] = vec2(1.0 / (first_column[0].x), 0.0);
	vec2 scaled_center;
	vec2 dot_product;
	float dot_sq;
	vec2 flipped_1;
	vec2 flipped_2;
	float factor;
	scaled_center = vec2(0.0, 0.0);
	dot_product = fma(out_solution[0].xx, first_column[1], scaled_center);
	dot_sq = cabs_sq(dot_product);
	factor = 1.0 / (1.0 - dot_sq);
	if(factor < 0.0) {
		dot_product = (one_minus_bias * inversesqrt(dot_sq)) * dot_product;
		first_column[1] = (dot_product - scaled_center) * (1.0 / out_solution[0].x);
		factor = corrected_factor;
		one_minus_bias = 0.0;
		corrected_factor = 1.0;
	}
	flipped_1 = vec2(out_solution[0].x, 0.0);
	out_solution[0] = vec2(factor * out_solution[0].x, 0.0);
	out_solution[1] = factor * (-flipped_1.x * dot_product);
	scaled_center = cmul(out_solution[1], first_column[1]);
	dot_product = fma(out_solution[0].xx, first_column[2], scaled_center);
	dot_sq = cabs_sq(dot_product);
	factor = 1.0 / (1.0 - dot_sq);
	if(factor < 0.0) {
		dot_product = (one_minus_bias * inversesqrt(dot_sq)) * dot_product;
		first_column[2] = (dot_product - scaled_center) * (1.0 / out_solution[0].x);
		factor = corrected_factor;
	}
	flipped_1 = cconj(out_solution[1]);
	flipped_2 = vec2(out_solution[0].x, 0.0);
	out_solution[0] = vec2(factor * out_solution[0].x, 0.0);
	out_solution[1] = factor * cfma(-flipped_1, dot_product, out_solution[1]);
	out_solution[2] = factor * (-flipped_2.x * dot_product);
}


//! Evaluates the autocorrelation of a complex signal with 3 entries and
//! outputs results for index shifts of 0, 1 or 2.
void real_autocorrelation_3(out vec2 out_autocorrelation[3], vec2 signal[3]) {
	out_autocorrelation[0] = cfma(signal[0], cconj(signal[0]), cfma(signal[1], cconj(signal[1]), cmul(signal[2], cconj(signal[2]))));
	out_autocorrelation[1] = cfma(signal[0], cconj(signal[1]), cmul(signal[1], cconj(signal[2])));
	out_autocorrelation[2] = cmul(signal[0], cconj(signal[2]));
}


//! Evaluates the first sum in Eq. 10: https://doi.org/10.1145/3306346.3322964
vec3 imag_correlation_3(vec2 lhs[3], vec2 rhs[3]) {
    return vec3(
	    fma(lhs[0].x, rhs[0].y, fma(lhs[0].y, rhs[0].x, fma(lhs[1].x, rhs[1].y, fma(lhs[1].y, rhs[1].x, fma(lhs[2].x, rhs[2].y, lhs[2].y * rhs[2].x))))),
	    fma(lhs[1].x, rhs[0].y, fma(lhs[1].y, rhs[0].x, fma(lhs[2].x, rhs[1].y, lhs[2].y * rhs[1].x))),
	    fma(lhs[2].x, rhs[0].y, lhs[2].y * rhs[0].x)
	);
}


//! Evaluates a Fourier series that is known to take real values, given cosine
//! and sine of the evaluation point and real Fourier coefficients 0, 1 and 2.
float eval_fourier_series_real_3(vec2 point, vec3 fouriers) {
    float cos_1 = point.x;
    float cos_2 = fma(point.x, point.x,-point.y * point.y);
    return 2.0 * fma(fouriers[1], cos_1, fma(fouriers[2], cos_2, 0.5 * fouriers[0]));
}


//! Applies the linear transform that turns linear Fourier sRGB into Fourier
//! coefficients that can be fed to prep_reflectance_real_lagrange_biased_3().
vec3 fourier_srgb_to_fourier(vec3 fourier_srgb) {
    // For easier portability, avoid working with builtin matrix types
    return vec3(
        dot(vec3(0.2276800310, 0.4748793271, 0.2993498525), fourier_srgb),
        dot(vec3(0.2035160895, 0.0770505049,-0.2808208130), fourier_srgb),
        dot(vec3(0.1563903497,-0.3230828819, 0.1668540863), fourier_srgb)
    );
}


/*! Prepares evaluation of a reflectance spectrum at specific wavelengths.
    \param trig_moments Three real, bounded trigonometric moments, i.e.
        Fourier coefficients of a reflectance spectrum. They may get modified
		slightly by the biasing procedure, if they would be invalid otherwise.
    \return Three Lagrange multipliers that should be fed into
        eval_reflectance_real_lagrange_3() to evaluate the spectrum.
    \note Implements the algorithm described at the end of Sec. 3.6.
        https://doi.org/10.1145/3306346.3322964 */
vec3 prep_reflectance_real_lagrange_biased_3(inout vec3 trig_moments) {
	trig_moments[0] = clamp(trig_moments[0], 0.0001, 0.9999);
	vec2 exp_moments[3];
	trig_to_exp_moments_real_3(exp_moments, trig_moments);
	vec2 eval_poly[3];
	levinson_3_biased(eval_poly, exp_moments);
	eval_poly[0] *= 6.28318531;
	eval_poly[1] *= 6.28318531;
	eval_poly[2] *= 6.28318531;
	vec2 autocorrelation[3];
	real_autocorrelation_3(autocorrelation, eval_poly);
	exp_moments[0] *= 0.5;
	float normalization_factor = 1.0 / (3.14159265 * eval_poly[0].x);
	return normalization_factor * imag_correlation_3(autocorrelation, exp_moments);
}


/*! Evaluates a reflectance spectrum at the given phase (which is a warped
    version of the wavelength) given Lagrange multipliers from
    prep_reflectance_real_lagrange_biased_3().*/
float eval_reflectance_real_lagrange_3(float phase, vec3 lagranges) {
	vec2 conj_circle_point = vec2(cos(-phase), sin(-phase));
	float lagrange_series = eval_fourier_series_real_3(conj_circle_point, lagranges);
	return fma(atan(lagrange_series), 0.318309886, 0.5);
}
