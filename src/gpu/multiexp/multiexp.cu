#ifndef COMMON_CL
#define COMMON_CL

// https://stackoverflow.com/questions/7696230/nvidia-nvcc-and-cuda-cubin-vs-ptx

////////////////////////////////////////////////////////////////////////
// Defines for CUDA
////////////////////////////////////////////////////////////////////////

extern "C" {

// inline -> __inline__
// __kernel -> __global__
// __global -> nothing
// __local -> TBD
#define DEVICE __device__
#define KERNEL __global__
#define GLOBAL

#define NVIDIA
typedef unsigned char uchar;
#define get_global_id(zz) blockIdx.x * blockDim.x + threadIdx.x

////////////////////////////////////////////////////////////////////////
// End defines for CUDA
////////////////////////////////////////////////////////////////////////


#ifdef __NV_CL_C_VERSION
#define NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #ifdef NVIDIA
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #ifdef NVIDIA
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #ifdef NVIDIA
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

#endif

#define Fr_limb ulong
#define Fr_LIMBS 4
#define Fr_LIMB_BITS 64
#define Fr_ONE ((Fr){ { 8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911 } })
#define Fr_P ((Fr){ { 18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352 } })
#define Fr_R2 ((Fr){ { 14526898881837571181, 3129137299524312099, 419701826671360399, 524908885293268753 } })
#define Fr_ZERO ((Fr){ { 0, 0, 0, 0 } })
#define Fr_INV 18446744069414584319
typedef struct { Fr_limb val[Fr_LIMBS]; } Fr;
#ifdef NVIDIA
DEVICE Fr Fr_sub_nvidia(Fr a, Fr b) {
asm("sub.cc.u64 %0, %0, %4;\r\n"
"subc.cc.u64 %1, %1, %5;\r\n"
"subc.cc.u64 %2, %2, %6;\r\n"
"subc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
DEVICE Fr Fr_add_nvidia(Fr a, Fr b) {
asm("add.cc.u64 %0, %0, %4;\r\n"
"addc.cc.u64 %1, %1, %5;\r\n"
"addc.cc.u64 %2, %2, %6;\r\n"
"addc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define Fr_BITS (Fr_LIMBS * Fr_LIMB_BITS)
#if Fr_LIMB_BITS == 32
  #define Fr_mac_with_carry mac_with_carry_32
  #define Fr_add_with_carry add_with_carry_32
#elif Fr_LIMB_BITS == 64
  #define Fr_mac_with_carry mac_with_carry_64
  #define Fr_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool Fr_gte(Fr a, Fr b) {
  for(char i = Fr_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool Fr_eq(Fr a, Fr b) {
  for(uchar i = 0; i < Fr_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#ifdef NVIDIA
  #define Fr_add_ Fr_add_nvidia
  #define Fr_sub_ Fr_sub_nvidia
#else
  DEVICE Fr Fr_add_(Fr a, Fr b) {
    bool carry = 0;
    for(uchar i = 0; i < Fr_LIMBS; i++) {
      Fr_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  DEVICE Fr Fr_sub_(Fr a, Fr b) {
    bool borrow = 0;
    for(uchar i = 0; i < Fr_LIMBS; i++) {
      Fr_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE Fr Fr_sub(Fr a, Fr b) {
  Fr res = Fr_sub_(a, b);
  if(!Fr_gte(a, b)) res = Fr_add_(res, Fr_P);
  return res;
}

// Modular addition
DEVICE Fr Fr_add(Fr a, Fr b) {
  Fr res = Fr_add_(a, b);
  if(Fr_gte(res, Fr_P)) res = Fr_sub_(res, Fr_P);
  return res;
}

// Modular multiplication
DEVICE Fr Fr_mul(Fr a, Fr b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  Fr_limb t[Fr_LIMBS + 2] = {0};
  for(uchar i = 0; i < Fr_LIMBS; i++) {
    Fr_limb carry = 0;
    for(uchar j = 0; j < Fr_LIMBS; j++)
      t[j] = Fr_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[Fr_LIMBS] = Fr_add_with_carry(t[Fr_LIMBS], &carry);
    t[Fr_LIMBS + 1] = carry;

    carry = 0;
    Fr_limb m = Fr_INV * t[0];
    Fr_mac_with_carry(m, Fr_P.val[0], t[0], &carry);
    for(uchar j = 1; j < Fr_LIMBS; j++)
      t[j - 1] = Fr_mac_with_carry(m, Fr_P.val[j], t[j], &carry);

    t[Fr_LIMBS - 1] = Fr_add_with_carry(t[Fr_LIMBS], &carry);
    t[Fr_LIMBS] = t[Fr_LIMBS + 1] + carry;
  }

  Fr result;
  for(uchar i = 0; i < Fr_LIMBS; i++) result.val[i] = t[i];

  if(Fr_gte(result, Fr_P)) result = Fr_sub_(result, Fr_P);

  return result;
}

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Fr Fr_sqr(Fr a) {
  return Fr_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Fr_add(a, a)
DEVICE Fr Fr_double(Fr a) {
  for(uchar i = Fr_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Fr_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(Fr_gte(a, Fr_P)) a = Fr_sub_(a, Fr_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE Fr Fr_pow(Fr base, uint exponent) {
  Fr res = Fr_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = Fr_mul(res, base);
    exponent = exponent >> 1;
    base = Fr_sqr(base);
  }
  return res;
}

// TODO: FFT
#ifdef zero

// Store squares of the base in a lookup table for faster evaluation.
DEVICE Fr Fr_pow_lookup(__global Fr *bases, uint exponent) {
  Fr res = Fr_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = Fr_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE Fr Fr_mont(Fr a) {
  return Fr_mul(a, Fr_R2);
}

DEVICE Fr Fr_unmont(Fr a) {
  Fr one = Fr_ZERO;
  one.val[0] = 1;
  return Fr_mul(a, one);
}

#endif // zero

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool Fr_get_bit(Fr l, uint i) {
  return (l.val[Fr_LIMBS - 1 - i / Fr_LIMB_BITS] >> (Fr_LIMB_BITS - 1 - (i % Fr_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint Fr_get_bits(Fr l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= Fr_get_bit(l, skip + i);
  }
  return ret;
}

DEVICE void Fr_print(Fr a) {
  printf("0x");
  for (uint i = 0; i < Fr_LIMBS; i++) {
    printf("%016lx", a.val[Fr_LIMBS - i - 1]);
  }
}

// TODO: FFT
#ifdef zero


DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
__kernel void radix_fft(__global Fr* x, // Source buffer
                        __global Fr* y, // Destination buffer
                        __global Fr* pq, // Precalculated twiddle factors
                        __global Fr* omegas, // [omega, omega^2, omega^4, ...]
                        __local Fr* u, // Local buffer to store intermediary values
                        uint n, // Number of elements
                        uint lgp, // Log2 of `p` (Read more in the link above)
                        uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);
  uint index = get_group_id(0);
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const Fr twiddle = Fr_pow_lookup(omegas, (n >> lgp >> deg) * k);
  Fr tmp = Fr_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = Fr_mul(tmp, x[i*t]);
    tmp = Fr_mul(tmp, twiddle);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = Fr_add(u[i0], u[i1]);
      u[i1] = Fr_sub(tmp, u[i1]);
      if(di != 0) u[i1] = Fr_mul(pq[di << rnd << pqshift], u[i1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
__kernel void mul_by_field(__global Fr* elements,
                        uint n,
                        Fr field) {
  const uint gid = get_global_id(0);
  elements[gid] = Fr_mul(elements[gid], field);
}
#endif // zero


#ifndef COMMON_CL
#define COMMON_CL

#ifdef __NV_CL_C_VERSION
#define NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #ifdef NVIDIA
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #ifdef NVIDIA
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #ifdef NVIDIA
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

#endif

#define Fq_limb ulong
#define Fq_LIMBS 6
#define Fq_LIMB_BITS 64
#define Fq_ONE ((Fq){ { 8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861, 6631298214892334189, 1582556514881692819 } })
#define Fq_P ((Fq){ { 13402431016077863595, 2210141511517208575, 7435674573564081700, 7239337960414712511, 5412103778470702295, 1873798617647539866 } })
#define Fq_R2 ((Fq){ { 17644856173732828998, 754043588434789617, 10224657059481499349, 7488229067341005760, 11130996698012816685, 1267921511277847466 } })
#define Fq_ZERO ((Fq){ { 0, 0, 0, 0, 0, 0 } })
#define Fq_INV 9940570264628428797
typedef struct { Fq_limb val[Fq_LIMBS]; } Fq;
#ifdef NVIDIA
DEVICE Fq Fq_sub_nvidia(Fq a, Fq b) {
asm("sub.cc.u64 %0, %0, %6;\r\n"
"subc.cc.u64 %1, %1, %7;\r\n"
"subc.cc.u64 %2, %2, %8;\r\n"
"subc.cc.u64 %3, %3, %9;\r\n"
"subc.cc.u64 %4, %4, %10;\r\n"
"subc.u64 %5, %5, %11;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
return a;
}
DEVICE Fq Fq_add_nvidia(Fq a, Fq b) {
asm("add.cc.u64 %0, %0, %6;\r\n"
"addc.cc.u64 %1, %1, %7;\r\n"
"addc.cc.u64 %2, %2, %8;\r\n"
"addc.cc.u64 %3, %3, %9;\r\n"
"addc.cc.u64 %4, %4, %10;\r\n"
"addc.u64 %5, %5, %11;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define Fq_BITS (Fq_LIMBS * Fq_LIMB_BITS)
#if Fq_LIMB_BITS == 32
  #define Fq_mac_with_carry mac_with_carry_32
  #define Fq_add_with_carry add_with_carry_32
#elif Fq_LIMB_BITS == 64
  #define Fq_mac_with_carry mac_with_carry_64
  #define Fq_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool Fq_gte(Fq a, Fq b) {
  for(char i = Fq_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool Fq_eq(Fq a, Fq b) {
  for(uchar i = 0; i < Fq_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#ifdef NVIDIA
  #define Fq_add_ Fq_add_nvidia
  #define Fq_sub_ Fq_sub_nvidia
#else
  DEVICE Fq Fq_add_(Fq a, Fq b) {
    bool carry = 0;
    for(uchar i = 0; i < Fq_LIMBS; i++) {
      Fq_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  DEVICE Fq Fq_sub_(Fq a, Fq b) {
    bool borrow = 0;
    for(uchar i = 0; i < Fq_LIMBS; i++) {
      Fq_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE Fq Fq_sub(Fq a, Fq b) {
  Fq res = Fq_sub_(a, b);
  if(!Fq_gte(a, b)) res = Fq_add_(res, Fq_P);
  return res;
}

// Modular addition
DEVICE Fq Fq_add(Fq a, Fq b) {
  Fq res = Fq_add_(a, b);
  if(Fq_gte(res, Fq_P)) res = Fq_sub_(res, Fq_P);
  return res;
}

// Modular multiplication
DEVICE Fq Fq_mul(Fq a, Fq b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  Fq_limb t[Fq_LIMBS + 2] = {0};
  for(uchar i = 0; i < Fq_LIMBS; i++) {
    Fq_limb carry = 0;
    for(uchar j = 0; j < Fq_LIMBS; j++)
      t[j] = Fq_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[Fq_LIMBS] = Fq_add_with_carry(t[Fq_LIMBS], &carry);
    t[Fq_LIMBS + 1] = carry;

    carry = 0;
    Fq_limb m = Fq_INV * t[0];
    Fq_mac_with_carry(m, Fq_P.val[0], t[0], &carry);
    for(uchar j = 1; j < Fq_LIMBS; j++)
      t[j - 1] = Fq_mac_with_carry(m, Fq_P.val[j], t[j], &carry);

    t[Fq_LIMBS - 1] = Fq_add_with_carry(t[Fq_LIMBS], &carry);
    t[Fq_LIMBS] = t[Fq_LIMBS + 1] + carry;
  }

  Fq result;
  for(uchar i = 0; i < Fq_LIMBS; i++) result.val[i] = t[i];

  if(Fq_gte(result, Fq_P)) result = Fq_sub_(result, Fq_P);

  return result;
}

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Fq Fq_sqr(Fq a) {
  return Fq_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Fq_add(a, a)
DEVICE Fq Fq_double(Fq a) {
  for(uchar i = Fq_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Fq_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(Fq_gte(a, Fq_P)) a = Fq_sub_(a, Fq_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE Fq Fq_pow(Fq base, uint exponent) {
  Fq res = Fq_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = Fq_mul(res, base);
    exponent = exponent >> 1;
    base = Fq_sqr(base);
  }
  return res;
}


// TODO: FFT
#ifdef zero

// Store squares of the base in a lookup table for faster evaluation.
DEVICE Fq Fq_pow_lookup(__global Fq *bases, uint exponent) {
  Fq res = Fq_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = Fq_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE Fq Fq_mont(Fq a) {
  return Fq_mul(a, Fq_R2);
}

DEVICE Fq Fq_unmont(Fq a) {
  Fq one = Fq_ZERO;
  one.val[0] = 1;
  return Fq_mul(a, one);
}
#endif // zero

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool Fq_get_bit(Fq l, uint i) {
  return (l.val[Fq_LIMBS - 1 - i / Fq_LIMB_BITS] >> (Fq_LIMB_BITS - 1 - (i % Fq_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint Fq_get_bits(Fq l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= Fq_get_bit(l, skip + i);
  }
  return ret;
}

DEVICE void Fq_print(Fq a) {
  printf("0x");
  for (uint i = 0; i < Fq_LIMBS; i++) {
    printf("%016lx", a.val[Fq_LIMBS - i - 1]);
  }
}


// Elliptic curve operations (Short Weierstrass Jacobian form)

#define G1_ZERO ((G1_projective){Fq_ZERO, Fq_ONE, Fq_ZERO})

 // Affine points in `blstrs` library do not have `inf` field.

typedef struct {
  Fq x;
  Fq y;
  #ifndef BLSTRS
    bool inf;
  #endif
  #if Fq_LIMB_BITS == 32
    uint _padding;
  #endif
} G1_affine;

typedef struct {
  Fq x;
  Fq y;
  Fq z;
} G1_projective;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE G1_projective G1_double(G1_projective inp) {
  const Fq local_zero = Fq_ZERO;
  if(Fq_eq(inp.z, local_zero)) {
      return inp;
  }

  const Fq a = Fq_sqr(inp.x); // A = X1^2
  const Fq b = Fq_sqr(inp.y); // B = Y1^2
  Fq c = Fq_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  Fq d = Fq_add(inp.x, b);
  d = Fq_sqr(d); d = Fq_sub(Fq_sub(d, a), c); d = Fq_double(d);

  const Fq e = Fq_add(Fq_double(a), a); // E = 3*A
  const Fq f = Fq_sqr(e);

  inp.z = Fq_mul(inp.y, inp.z); inp.z = Fq_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = Fq_sub(Fq_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = Fq_double(c); c = Fq_double(c); c = Fq_double(c);
  inp.y = Fq_sub(Fq_mul(Fq_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE G1_projective G1_add_mixed(G1_projective a, G1_affine b) {
  #ifndef BLSTRS
    if(b.inf) {
        return a;
    }
  #endif

  const Fq local_zero = Fq_ZERO;
  if(Fq_eq(a.z, local_zero)) {
    const Fq local_one = Fq_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const Fq z1z1 = Fq_sqr(a.z);
  const Fq u2 = Fq_mul(b.x, z1z1);
  const Fq s2 = Fq_mul(Fq_mul(b.y, a.z), z1z1);

  if(Fq_eq(a.x, u2) && Fq_eq(a.y, s2)) {
      return G1_double(a);
  }

  const Fq h = Fq_sub(u2, a.x); // H = U2-X1
  const Fq hh = Fq_sqr(h); // HH = H^2
  Fq i = Fq_double(hh); i = Fq_double(i); // I = 4*HH
  Fq j = Fq_mul(h, i); // J = H*I
  Fq r = Fq_sub(s2, a.y); r = Fq_double(r); // r = 2*(S2-Y1)
  const Fq v = Fq_mul(a.x, i);

  G1_projective ret;

  // X3 = r^2 - J - 2*V
  ret.x = Fq_sub(Fq_sub(Fq_sqr(r), j), Fq_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = Fq_mul(a.y, j); j = Fq_double(j);
  ret.y = Fq_sub(Fq_mul(Fq_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = Fq_add(a.z, h); ret.z = Fq_sub(Fq_sub(Fq_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE G1_projective G1_add(G1_projective a, G1_projective b) {

  const Fq local_zero = Fq_ZERO;
  if(Fq_eq(a.z, local_zero)) return b;
  if(Fq_eq(b.z, local_zero)) return a;

  const Fq z1z1 = Fq_sqr(a.z); // Z1Z1 = Z1^2
  const Fq z2z2 = Fq_sqr(b.z); // Z2Z2 = Z2^2
  const Fq u1 = Fq_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const Fq u2 = Fq_mul(b.x, z1z1); // U2 = X2*Z1Z1
  Fq s1 = Fq_mul(Fq_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const Fq s2 = Fq_mul(Fq_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(Fq_eq(u1, u2) && Fq_eq(s1, s2))
    return G1_double(a);
  else {
    const Fq h = Fq_sub(u2, u1); // H = U2-U1
    Fq i = Fq_double(h); i = Fq_sqr(i); // I = (2*H)^2
    const Fq j = Fq_mul(h, i); // J = H*I
    Fq r = Fq_sub(s2, s1); r = Fq_double(r); // r = 2*(S2-S1)
    const Fq v = Fq_mul(u1, i); // V = U1*I
    a.x = Fq_sub(Fq_sub(Fq_sub(Fq_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = Fq_mul(Fq_sub(v, a.x), r);
    s1 = Fq_mul(s1, j); s1 = Fq_double(s1); // S1 = S1 * J * 2
    a.y = Fq_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = Fq_add(a.z, b.z); a.z = Fq_sqr(a.z);
    a.z = Fq_sub(Fq_sub(a.z, z1z1), z2z2);
    a.z = Fq_mul(a.z, h);

    return a;
  }
}


/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void g1_bellman_multiexp(
    GLOBAL G1_affine *bases,
    GLOBAL G1_projective *buckets,
    GLOBAL G1_projective *results,
    GLOBAL Fr *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = get_global_id(0);
  if(gid >= num_windows * num_groups) return;

  // if(gid == 0) {
  //   printf("gid 0\n");
  //   printf("  sizeof G1_affine is %ld\n", sizeof(G1_affine));
  //   printf("  bases       %p\n", bases);
  //   printf("  buckets     %p\n", buckets);
  //   printf("  results     %p\n", results);
  //   printf("  exps        %p\n", exps);
  //   printf("  n           %d\n", n);
  //   printf("  num_groups  %d\n", num_groups);
  //   printf("  num_windows %d\n", num_windows);
  //   printf("  window_size %d\n", window_size);
  // }
  
  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const G1_projective local_zero = G1_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  const uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(Fr_BITS - bits));

  G1_projective res = G1_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = Fr_get_bits(exps[i], bits, w);

    #ifdef NVIDIA
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = G1_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = G1_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = G1_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  G1_projective acc = G1_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = G1_add(acc, buckets[j]);
    res = G1_add(res, acc);
  }

  results[gid] = res;
}


// Fp2 Extension Field where u^2 + 1 = 0

#define Fq2_LIMB_BITS Fq_LIMB_BITS
#define Fq2_ZERO ((Fq2){Fq_ZERO, Fq_ZERO})
#define Fq2_ONE ((Fq2){Fq_ONE, Fq_ZERO})

typedef struct {
  Fq c0;
  Fq c1;
} Fq2; // Represents: c0 + u * c1

DEVICE bool Fq2_eq(Fq2 a, Fq2 b) {
  return Fq_eq(a.c0, b.c0) && Fq_eq(a.c1, b.c1);
}
DEVICE Fq2 Fq2_sub(Fq2 a, Fq2 b) {
  a.c0 = Fq_sub(a.c0, b.c0);
  a.c1 = Fq_sub(a.c1, b.c1);
  return a;
}
DEVICE Fq2 Fq2_add(Fq2 a, Fq2 b) {
  a.c0 = Fq_add(a.c0, b.c0);
  a.c1 = Fq_add(a.c1, b.c1);
  return a;
}
DEVICE Fq2 Fq2_double(Fq2 a) {
  a.c0 = Fq_double(a.c0);
  a.c1 = Fq_double(a.c1);
  return a;
}

/*
 * (a_0 + u * a_1)(b_0 + u * b_1) = a_0 * b_0 - a_1 * b_1 + u * (a_0 * b_1 + a_1 * b_0)
 * Therefore:
 * c_0 = a_0 * b_0 - a_1 * b_1
 * c_1 = (a_0 * b_1 + a_1 * b_0) = (a_0 + a_1) * (b_0 + b_1) - a_0 * b_0 - a_1 * b_1
 */
DEVICE Fq2 Fq2_mul(Fq2 a, Fq2 b) {
  const Fq aa = Fq_mul(a.c0, b.c0);
  const Fq bb = Fq_mul(a.c1, b.c1);
  const Fq o = Fq_add(b.c0, b.c1);
  a.c1 = Fq_add(a.c1, a.c0);
  a.c1 = Fq_mul(a.c1, o);
  a.c1 = Fq_sub(a.c1, aa);
  a.c1 = Fq_sub(a.c1, bb);
  a.c0 = Fq_sub(aa, bb);
  return a;
}

/*
 * (a_0 + u * a_1)(a_0 + u * a_1) = a_0 ^ 2 - a_1 ^ 2 + u * 2 * a_0 * a_1
 * Therefore:
 * c_0 = (a_0 * a_0 - a_1 * a_1) = (a_0 + a_1)(a_0 - a_1)
 * c_1 = 2 * a_0 * a_1
 */
DEVICE Fq2 Fq2_sqr(Fq2 a) {
  const Fq ab = Fq_mul(a.c0, a.c1);
  const Fq c0c1 = Fq_add(a.c0, a.c1);
  a.c0 = Fq_mul(Fq_sub(a.c0, a.c1), c0c1);
  a.c1 = Fq_double(ab);
  return a;
}


// Elliptic curve operations (Short Weierstrass Jacobian form)

#define G2_ZERO ((G2_projective){Fq2_ZERO, Fq2_ONE, Fq2_ZERO})

 // Affine points in `blstrs` library do not have `inf` field.

typedef struct {
  Fq2 x;
  Fq2 y;
  #ifndef BLSTRS
    bool inf;
  #endif
  #if Fq2_LIMB_BITS == 32
    uint _padding;
  #endif
} G2_affine;

typedef struct {
  Fq2 x;
  Fq2 y;
  Fq2 z;
} G2_projective;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE G2_projective G2_double(G2_projective inp) {
  const Fq2 local_zero = Fq2_ZERO;
  if(Fq2_eq(inp.z, local_zero)) {
      return inp;
  }

  const Fq2 a = Fq2_sqr(inp.x); // A = X1^2
  const Fq2 b = Fq2_sqr(inp.y); // B = Y1^2
  Fq2 c = Fq2_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  Fq2 d = Fq2_add(inp.x, b);
  d = Fq2_sqr(d); d = Fq2_sub(Fq2_sub(d, a), c); d = Fq2_double(d);

  const Fq2 e = Fq2_add(Fq2_double(a), a); // E = 3*A
  const Fq2 f = Fq2_sqr(e);

  inp.z = Fq2_mul(inp.y, inp.z); inp.z = Fq2_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = Fq2_sub(Fq2_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = Fq2_double(c); c = Fq2_double(c); c = Fq2_double(c);
  inp.y = Fq2_sub(Fq2_mul(Fq2_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE G2_projective G2_add_mixed(G2_projective a, G2_affine b) {
  #ifndef BLSTRS
    if(b.inf) {
        return a;
    }
  #endif

  const Fq2 local_zero = Fq2_ZERO;
  if(Fq2_eq(a.z, local_zero)) {
    const Fq2 local_one = Fq2_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const Fq2 z1z1 = Fq2_sqr(a.z);
  const Fq2 u2 = Fq2_mul(b.x, z1z1);
  const Fq2 s2 = Fq2_mul(Fq2_mul(b.y, a.z), z1z1);

  if(Fq2_eq(a.x, u2) && Fq2_eq(a.y, s2)) {
      return G2_double(a);
  }

  const Fq2 h = Fq2_sub(u2, a.x); // H = U2-X1
  const Fq2 hh = Fq2_sqr(h); // HH = H^2
  Fq2 i = Fq2_double(hh); i = Fq2_double(i); // I = 4*HH
  Fq2 j = Fq2_mul(h, i); // J = H*I
  Fq2 r = Fq2_sub(s2, a.y); r = Fq2_double(r); // r = 2*(S2-Y1)
  const Fq2 v = Fq2_mul(a.x, i);

  G2_projective ret;

  // X3 = r^2 - J - 2*V
  ret.x = Fq2_sub(Fq2_sub(Fq2_sqr(r), j), Fq2_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = Fq2_mul(a.y, j); j = Fq2_double(j);
  ret.y = Fq2_sub(Fq2_mul(Fq2_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = Fq2_add(a.z, h); ret.z = Fq2_sub(Fq2_sub(Fq2_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE G2_projective G2_add(G2_projective a, G2_projective b) {

  const Fq2 local_zero = Fq2_ZERO;
  if(Fq2_eq(a.z, local_zero)) return b;
  if(Fq2_eq(b.z, local_zero)) return a;

  const Fq2 z1z1 = Fq2_sqr(a.z); // Z1Z1 = Z1^2
  const Fq2 z2z2 = Fq2_sqr(b.z); // Z2Z2 = Z2^2
  const Fq2 u1 = Fq2_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const Fq2 u2 = Fq2_mul(b.x, z1z1); // U2 = X2*Z1Z1
  Fq2 s1 = Fq2_mul(Fq2_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const Fq2 s2 = Fq2_mul(Fq2_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(Fq2_eq(u1, u2) && Fq2_eq(s1, s2))
    return G2_double(a);
  else {
    const Fq2 h = Fq2_sub(u2, u1); // H = U2-U1
    Fq2 i = Fq2_double(h); i = Fq2_sqr(i); // I = (2*H)^2
    const Fq2 j = Fq2_mul(h, i); // J = H*I
    Fq2 r = Fq2_sub(s2, s1); r = Fq2_double(r); // r = 2*(S2-S1)
    const Fq2 v = Fq2_mul(u1, i); // V = U1*I
    a.x = Fq2_sub(Fq2_sub(Fq2_sub(Fq2_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = Fq2_mul(Fq2_sub(v, a.x), r);
    s1 = Fq2_mul(s1, j); s1 = Fq2_double(s1); // S1 = S1 * J * 2
    a.y = Fq2_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = Fq2_add(a.z, b.z); a.z = Fq2_sqr(a.z);
    a.z = Fq2_sub(Fq2_sub(a.z, z1z1), z2z2);
    a.z = Fq2_mul(a.z, h);

    return a;
  }
}


/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void g2_bellman_multiexp(
    GLOBAL G2_affine *bases,
    GLOBAL G2_projective *buckets,
    GLOBAL G2_projective *results,
    GLOBAL Fr *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = get_global_id(0);
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const G2_projective local_zero = G2_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  const uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(Fr_BITS - bits));

  G2_projective res = G2_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = Fr_get_bits(exps[i], bits, w);

    #ifdef NVIDIA
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = G2_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = G2_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = G2_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  G2_projective acc = G2_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = G2_add(acc, buckets[j]);
    res = G2_add(res, acc);
  }

  results[gid] = res;
}

}
