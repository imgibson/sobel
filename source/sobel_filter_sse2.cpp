/*!
 * Sobel Filter (the "software") provided by Anders Lind ("author") license agreements.
 * - This software is free for both personal and commercial use. You may install and use it on your computers free of charge.
 * - You may NOT modify, de-compile, disassemble or reverse engineer the software.
 * - You may use, copy, sell, redistribute or give the software to third part freely as long as the software is not modified.
 * - The software remains property of the authors also in case of dissemination to third parties.
 * - The software's name and logo are not to be used to identify other products or services.
 * - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * - The authors reserve the rights to change the license agreements in future versions of the software
 */

#include "sobel_filter.h"

#include <cstdint>
#include <cassert>
#include <cmath>

#include <emmintrin.h> // Intel SSE2

#define RSHIFT(shift) _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(shift), 4))
#define LSHIFT(shift) _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(shift), 4))
#define RSHIFTM(shift, merge) _mm_or_ps(RSHIFT(shift), _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(merge), 12)))
#define LSHIFTM(shift, merge) _mm_or_ps(LSHIFT(shift), _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(merge), 12)))

static const float ScaleFactor = 1.0f / sqrtf(32.0f);
static const __m128 ScaleVec = _mm_set1_ps(ScaleFactor);

static inline const float* offset_float_ptr(const float* ptr, uintptr_t byteOffset) { return reinterpret_cast<const float*>(&(reinterpret_cast<const uint8_t*>(ptr)[byteOffset])); }
static inline float* offset_float_ptr(float* ptr, uintptr_t byteOffset) { return reinterpret_cast<float*>(&(reinterpret_cast<uint8_t*>(ptr)[byteOffset])); }

void sobel_filter_sse2(const float* __restrict src, float* __restrict dst, uint32_t width, uint32_t height, uint32_t bytesPerLineSrc, uint32_t bytesPerLineDst)
{
	// Verify 128 bit alignment
	assert((reinterpret_cast<uintptr_t>(src) & 15u) == 0u);
	assert((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u);
	assert((bytesPerLineSrc & 15u) == 0u);
	assert((bytesPerLineDst & 15u) == 0u);
	// Verify minimum SIMD width
	assert(width >= 4u);

	const float* pr = src;
	const float* cr = src;
	const float* nr = offset_float_ptr(src, bytesPerLineSrc);
	const float* lr = offset_float_ptr(src, (height - 1u) * static_cast<uintptr_t>(bytesPerLineSrc));

	float* dr = dst;

	if ((width & 3u) == 0u) // Even mulitple of SIMD size
	{
		if (width >= 8u)
		{
			while (pr < lr)
			{
				__m128 top = _mm_load_ps(pr);
				__m128 mid = _mm_load_ps(cr);
				__m128 low = _mm_load_ps(nr);

				__m128 currx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
				__m128 curry = _mm_sub_ps(top, low);

				top = _mm_load_ps(&pr[4u]);
				mid = _mm_load_ps(&cr[4u]);
				low = _mm_load_ps(&nr[4u]);

				__m128 nextx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
				__m128 nexty = _mm_sub_ps(top, low);

				__m128 xout = _mm_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm_mul_ps(xout, xout);

				__m128 out = _mm_add_ps(_mm_add_ps(curry, curry), _mm_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm_mul_ps(out, out);
				out = _mm_add_ps(out, xout);

				out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

				_mm_store_ps(dr, out);

				uint32_t x = 8u;

				for (; x < width; x += 4u)
				{
					const __m128 prevx = currx;
					const __m128 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm_load_ps(&pr[x]);
					mid = _mm_load_ps(&cr[x]);
					low = _mm_load_ps(&nr[x]);

					nextx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
					nexty = _mm_sub_ps(top, low);

					xout = _mm_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm_mul_ps(xout, xout);

					out = _mm_add_ps(_mm_add_ps(curry, curry), _mm_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm_mul_ps(out, out);
					out = _mm_add_ps(out, xout);

					out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

					_mm_store_ps(&dr[x - 4u], out);
				}

				xout = _mm_sub_ps(RSHIFTM(nextx, currx), LSHIFT(nextx));
				xout = _mm_mul_ps(xout, xout);

				out = _mm_add_ps(_mm_add_ps(nexty, nexty), _mm_add_ps(RSHIFTM(nexty, curry), LSHIFT(nexty)));
				out = _mm_add_ps(_mm_mul_ps(out, out), xout);

				out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

				_mm_store_ps(&dr[x - 4u], out);

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytesPerLineSrc);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytesPerLineDst);
			}
		}
		else
		{
			while (pr < lr)
			{
				const __m128 top = _mm_load_ps(pr);
				const __m128 mid = _mm_load_ps(cr);
				const __m128 low = _mm_load_ps(nr);

				__m128 xout = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
				xout = _mm_sub_ps(RSHIFT(xout), LSHIFT(xout));
				xout = _mm_mul_ps(xout, xout);

				__m128 out = _mm_sub_ps(top, low);
				out = _mm_add_ps(_mm_add_ps(out, out), _mm_add_ps(RSHIFT(out), LSHIFT(out)));
				out = _mm_add_ps(_mm_mul_ps(out, out), xout);

				out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

				_mm_store_ps(dr, out);

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytesPerLineSrc);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytesPerLineDst);
			}
		}
	}
	else
	{
		if (width >= 8u)
		{
			const uint32_t count = 4u * (width / 4u);
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
				__m128 top = _mm_load_ps(pr);
				__m128 mid = _mm_load_ps(cr);
				__m128 low = _mm_load_ps(nr);

				__m128 currx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
				__m128 curry = _mm_sub_ps(top, low);

				top = _mm_load_ps(&pr[4u]);
				mid = _mm_load_ps(&cr[4u]);
				low = _mm_load_ps(&nr[4u]);

				__m128 nextx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
				__m128 nexty = _mm_sub_ps(top, low);

				__m128 xout = _mm_sub_ps(RSHIFT(currx), LSHIFTM(currx, nextx));
				xout = _mm_mul_ps(xout, xout);

				__m128 out = _mm_add_ps(_mm_add_ps(curry, curry), _mm_add_ps(RSHIFT(curry), LSHIFTM(curry, nexty)));
				out = _mm_add_ps(_mm_mul_ps(out, out), xout);

				out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

				_mm_store_ps(dr, out);

				uint32_t x = 8u;

				for (; x < count; x += 4u)
				{
					const __m128 prevx = currx;
					const __m128 prevy = curry;

					currx = nextx;
					curry = nexty;

					top = _mm_load_ps(&pr[x]);
					mid = _mm_load_ps(&cr[x]);
					low = _mm_load_ps(&nr[x]);

					nextx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
					nexty = _mm_sub_ps(top, low);

					xout = _mm_sub_ps(RSHIFTM(currx, prevx), LSHIFTM(currx, nextx));
					xout = _mm_mul_ps(xout, xout);

					out = _mm_add_ps(_mm_add_ps(curry, curry), _mm_add_ps(RSHIFTM(curry, prevy), LSHIFTM(curry, nexty)));
					out = _mm_add_ps(_mm_mul_ps(out, out), xout);

					out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

					_mm_store_ps(&dr[x - 4u], out);
				}

				for (; x < lx; ++x)
				{
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				{
					const float dx =
						1.0f * (pr[lx] - pr[lx - 1u]) +
						2.0f * (cr[lx] - cr[lx - 1u]) +
						1.0f * (nr[lx] - nr[lx - 1u]);

					const float dy =
						1.0f * (pr[lx - 1u] - nr[lx - 1u]) +
						2.0f * (pr[lx] - nr[lx]) +
						1.0f * (pr[lx] - nr[lx]);

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytesPerLineSrc);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytesPerLineDst);
			}
		}
		else
		{
			const uint32_t lx = width - 1u;

			while (pr < lr)
			{
				{
					const __m128 top = _mm_load_ps(pr);
					const __m128 mid = _mm_load_ps(cr);
					const __m128 low = _mm_load_ps(nr);

					const __m128 currx = _mm_add_ps(_mm_add_ps(mid, mid), _mm_add_ps(top, low));
					const __m128 curry = _mm_sub_ps(top, low);

					__m128 xout = _mm_sub_ps(RSHIFT(currx), LSHIFT(currx));
					xout = _mm_mul_ps(xout, xout);

					__m128 out = _mm_add_ps(_mm_add_ps(curry, curry), _mm_add_ps(RSHIFT(curry), LSHIFT(curry)));
					out = _mm_add_ps(_mm_mul_ps(out, out), xout);

					out = _mm_mul_ps(_mm_sqrt_ps(out), ScaleVec);

					_mm_store_ps(dr, out);
				}

				for (uint32_t x = 3u; x < lx; ++x)
				{
					const float dx =
						1.0f * (pr[x + 1u] - pr[x - 1u]) +
						2.0f * (cr[x + 1u] - cr[x - 1u]) +
						1.0f * (nr[x + 1u] - nr[x - 1u]);

					const float dy =
						1.0f * (pr[x - 1u] - nr[x - 1u]) +
						2.0f * (pr[x] - nr[x]) +
						1.0f * (pr[x + 1u] - nr[x + 1u]);

					dr[x] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				{
					const float dx =
						1.0f * (pr[lx] - pr[lx - 1u]) +
						2.0f * (cr[lx] - cr[lx - 1u]) +
						1.0f * (nr[lx] - nr[lx - 1u]);

					const float dy =
						1.0f * (pr[lx - 1u] - nr[lx - 1u]) +
						2.0f * (pr[lx] - nr[lx]) +
						1.0f * (pr[lx] - nr[lx]);

					dr[lx] = sqrtf(dx * dx + dy * dy) * ScaleFactor;
				}

				pr = cr;
				cr = nr;
				nr = offset_float_ptr(nr, bytesPerLineSrc);
				if (nr > lr)
					nr = lr;

				dr = offset_float_ptr(dr, bytesPerLineDst);
			}
		}
	}
}
