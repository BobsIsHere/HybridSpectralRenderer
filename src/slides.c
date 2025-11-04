#include "main.h"
#include "string_utilities.h"


uint32_t create_slides(slide_t* slides) {
	uint32_t n = 0;
	render_settings_t default_rgb = {
		.path_length = 4,
		.wavelength_sample_count = 4,
		.color_model = color_model_rgb,
	};
	render_settings_t default_spectral = {
		.path_length = 4,
		.wavelength_sample_count = 4,
		.color_model = color_model_spectral,
	};
	render_settings_t direct_rgb = {
		.path_length = 2,
		.wavelength_sample_count = 4,
		.color_model = color_model_rgb,
	};
	render_settings_t direct_spectral = {
		.path_length = 2,
		.wavelength_sample_count = 4,
		.color_model = color_model_spectral,
	};
	// Slides showing the entrance to the bistro with various illuminants
	uint32_t illuminants[] = { 2474, 2475, 2479, 2481, 2510, 2525, 2536, 2661 };
	for (uint32_t i = 0; i != COUNT_OF(illuminants); ++i) {
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance.rt_save"),
			.render_settings = default_rgb,
			.screenshot_frame = 2048,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_%u_rgb.png", illuminants[i]),
		};
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance.rt_save"),
			.render_settings = default_spectral,
			.screenshot_frame = 2048,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_%u_spectral.png", illuminants[i]),
		};
	}
	// Same thing with 1 spp
	for (uint32_t i = 0; i != COUNT_OF(illuminants); ++i) {
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance.rt_save"),
			.render_settings = default_rgb,
			.screenshot_frame = 1,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_1spp_%u_rgb.png", illuminants[i]),
		};
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance.rt_save"),
			.render_settings = default_spectral,
			.screenshot_frame = 1,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_1spp_%u_spectral.png", illuminants[i]),
		};
	}
	// Same thing using a single light source and a setup where the RGB
	// renderer is noise-free (when MIS, shadow rays and jittering of primary
	// rays are commented out in path_trace.frag.glsl)
	for (uint32_t i = 0; i != COUNT_OF(illuminants); ++i) {
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance_1_light.rt_save"),
			.render_settings = direct_rgb,
			.screenshot_frame = 1,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_1_light_%u_rgb.png", illuminants[i]),
		};
		slides[n++] = (slide_t) {
			.quicksave = copy_string("data/saves/bistro/entrance_1_light.rt_save"),
			.render_settings = direct_spectral,
			.screenshot_frame = 1,
			.emission_material_spectrum_id = illuminants[i],
			.screenshot_path = format_uint("data/slides/entrance_1_light_%u_spectral.png", illuminants[i]),
		};
	}
	printf("Defined %u slides.\n", n);
	if (n > MAX_SLIDE_COUNT)
		printf("WARNING: Wrote %u slides but MAX_SLIDE_COUNT is %u. Increase it.\n", n, MAX_SLIDE_COUNT);
	return n;
}
