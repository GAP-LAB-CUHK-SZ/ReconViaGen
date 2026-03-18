// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_FP16_RVV10_H
#define EIGEN_PACKET_MATH_FP16_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<vfloat16m1_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL))), 24>
    Packet1Xh;
typedef eigen_packet_wrapper<vfloat16m2_t __attribute__((riscv_rvv_vector_bits(EIGEN_RISCV64_RVV_VL * 2))), 25>
    Packet2Xh;

#if EIGEN_RISCV64_DEFAULT_LMUL == 1
typedef Packet1Xh PacketXh;

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet1Xh type;
  typedef Packet1Xh half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<Eigen::half, EIGEN_RISCV64_RVV_VL, 1>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 0,
    HasExp = 0,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = 0
  };
};

#else
typedef Packet2Xh PacketXh;

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet2Xh type;
  typedef Packet1Xh half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<Eigen::half, EIGEN_RISCV64_RVV_VL, 2>::size,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,

    HasCmp = 1,
    HasDiv = 1,
    HasRound = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 0,
    HasExp = 0,
    HasSqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = 0
  };
};
#endif

template <>
struct unpacket_traits<Packet1Xh> {
  typedef Eigen::half type;
  typedef Packet1Xh half;  // Half not yet implemented
  typedef PacketXs integer_packet;
  typedef numext::uint8_t mask_t;

  enum {
    size = rvv_packet_size_selector<Eigen::half, EIGEN_RISCV64_RVV_VL, 1>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 1>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet2Xh> {
  typedef Eigen::half type;
  typedef Packet1Xh half;
  typedef Packet2Xs integer_packet;
  typedef numext::uint8_t mask_t;

  enum {
    size = rvv_packet_size_selector<Eigen::half, EIGEN_RISCV64_RVV_VL, 2>::size,
    alignment = rvv_packet_alignment_selector<EIGEN_RISCV64_RVV_VL, 2>::alignment,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

/********************************* PacketXh ************************************/

template <>
EIGEN_STRONG_INLINE PacketXh ptrue<PacketXh>(const PacketXh& /*a*/) {
  return __riscv_vreinterpret_f16m1(__riscv_vmv_v_x_u16m1(0xffffu, unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh pzero<PacketXh>(const PacketXh& /*a*/) {
  return __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.0), unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pabs(const PacketXh& a) {
  return __riscv_vfabs_v_f16m1(a, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pset1<PacketXh>(const Eigen::half& from) {
  return __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(from), unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pset1frombits<PacketXh>(numext::uint16_t from) {
  return __riscv_vreinterpret_f16m1(__riscv_vmv_v_x_u16m1(from, unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh plset<PacketXh>(const Eigen::half& a) {
  PacketXh idx =
      __riscv_vfcvt_f_x_v_f16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketXs>::size)),
      unpacket_traits<PacketXh>::size);
  return __riscv_vfadd_vf_f16m1(idx, numext::bit_cast<_Float16>(a), unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh padd<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfadd_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh psub<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfsub_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pnegate(const PacketXh& a) {
  return __riscv_vfneg_v_f16m1(a, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pconj(const PacketXh& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXh pmul<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfmul_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pdiv<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfdiv_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmadd(const PacketXh& a, const PacketXh& b, const PacketXh& c) {
  return __riscv_vfmadd_vv_f16m1(a, b, c, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmsub(const PacketXh& a, const PacketXh& b, const PacketXh& c) {
  return __riscv_vfmsub_vv_f16m1(a, b, c, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pnmadd(const PacketXh& a, const PacketXh& b, const PacketXh& c) {
  return __riscv_vfnmsub_vv_f16m1(a, b, c, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pnmsub(const PacketXh& a, const PacketXh& b, const PacketXh& c) {
  return __riscv_vfnmadd_vv_f16m1(a, b, c, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmin<PacketXh>(const PacketXh& a, const PacketXh& b) {
  const Eigen::half nan = (std::numeric_limits<Eigen::half>::quiet_NaN)();
  PacketXh nans =
      __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(nan), unpacket_traits<PacketXh>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f16m1_b16(a, a, unpacket_traits<PacketXh>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f16m1_b16(b, b, unpacket_traits<PacketXh>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketXh>::size);

  return __riscv_vfmin_vv_f16m1_tumu(mask, nans, a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmin<PropagateNaN, PacketXh>(const PacketXh& a, const PacketXh& b) {
  return pmin<PacketXh>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmin<PropagateNumbers, PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfmin_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmax<PacketXh>(const PacketXh& a, const PacketXh& b) {
  const Eigen::half nan = (std::numeric_limits<Eigen::half>::quiet_NaN)();
  PacketXh nans =
      __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(nan), unpacket_traits<PacketXh>::size);
  PacketMask16 mask = __riscv_vmfeq_vv_f16m1_b16(a, a, unpacket_traits<PacketXh>::size);
  PacketMask16 mask2 = __riscv_vmfeq_vv_f16m1_b16(b, b, unpacket_traits<PacketXh>::size);
  mask = __riscv_vmand_mm_b16(mask, mask2, unpacket_traits<PacketXh>::size);

  return __riscv_vfmax_vv_f16m1_tumu(mask, nans, a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmax<PropagateNaN, PacketXh>(const PacketXh& a, const PacketXh& b) {
  return pmax<PacketXh>(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketXh pmax<PropagateNumbers, PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vfmax_vv_f16m1(a, b, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pcmp_le<PacketXh>(const PacketXh& a, const PacketXh& b) {
  PacketMask16 mask = __riscv_vmfle_vv_f16m1_b16(a, b, unpacket_traits<PacketXh>::size);
  return __riscv_vmerge_vvm_f16m1(pzero<PacketXh>(a), ptrue<PacketXh>(a), mask, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pcmp_lt<PacketXh>(const PacketXh& a, const PacketXh& b) {
  PacketMask16 mask = __riscv_vmflt_vv_f16m1_b16(a, b, unpacket_traits<PacketXh>::size);
  return __riscv_vmerge_vvm_f16m1(pzero<PacketXh>(a), ptrue<PacketXh>(a), mask, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pcmp_eq<PacketXh>(const PacketXh& a, const PacketXh& b) {
  PacketMask16 mask = __riscv_vmfeq_vv_f16m1_b16(a, b, unpacket_traits<PacketXh>::size);
  return __riscv_vmerge_vvm_f16m1(pzero<PacketXh>(a), ptrue<PacketXh>(a), mask, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pcmp_lt_or_nan<PacketXh>(const PacketXh& a, const PacketXh& b) {
  PacketMask16 mask = __riscv_vmfge_vv_f16m1_b16(a, b, unpacket_traits<PacketXh>::size);
  return __riscv_vfmerge_vfm_f16m1(ptrue<PacketXh>(a), static_cast<_Float16>(0.0), mask,
                                   unpacket_traits<PacketXh>::size);
}

// Logical Operations are not supported for half, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketXh pand<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(
      __riscv_vreinterpret_v_f16m1_u16m1(a), __riscv_vreinterpret_v_f16m1_u16m1(b), unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh por<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(
      __riscv_vreinterpret_v_f16m1_u16m1(a), __riscv_vreinterpret_v_f16m1_u16m1(b), unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh pxor<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vxor_vv_u16m1(
      __riscv_vreinterpret_v_f16m1_u16m1(a), __riscv_vreinterpret_v_f16m1_u16m1(b), unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh pandnot<PacketXh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(
      __riscv_vreinterpret_v_f16m1_u16m1(a),
      __riscv_vnot_v_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(b), unpacket_traits<PacketXh>::size),
      unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE PacketXh pload<PacketXh>(const Eigen::half* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_f16m1(reinterpret_cast<const _Float16*>(from),
                                                        unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh ploadu<PacketXh>(const Eigen::half* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_f16m1(reinterpret_cast<const _Float16*>(from),
                                                          unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh ploaddup<PacketXh>(const Eigen::half* from) {
  PacketXsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketXh>::size);
  idx = __riscv_vand_vx_u16m1(idx, 0xfffeu, unpacket_traits<PacketXh>::size);
  return __riscv_vloxei16_v_f16m1(reinterpret_cast<const _Float16*>(from), idx, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh ploadquad<PacketXh>(const Eigen::half* from) {
  PacketXsu idx = __riscv_vid_v_u16m1(unpacket_traits<PacketXh>::size);
  idx = __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(idx, 0xfffcu, unpacket_traits<PacketXh>::size), 1,
                              unpacket_traits<PacketXh>::size);
  return __riscv_vloxei16_v_f16m1(reinterpret_cast<const _Float16*>(from), idx, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const PacketXh& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_f16m1(reinterpret_cast<_Float16*>(to), from,
                                                  unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const PacketXh& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_f16m1(reinterpret_cast<_Float16*>(to), from,
                                                    unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXh pgather<Eigen::half, PacketXh>(const Eigen::half* from, Index stride) {
  return __riscv_vlse16_v_f16m1(reinterpret_cast<const _Float16*>(from), stride * sizeof(Eigen::half),
                                unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<Eigen::half, PacketXh>(Eigen::half* to, const PacketXh& from, Index stride) {
  __riscv_vsse16(reinterpret_cast<_Float16*>(to), stride * sizeof(Eigen::half), from, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE Eigen::half pfirst<PacketXh>(const PacketXh& a) {
  return static_cast<Eigen::half>(__riscv_vfmv_f_s_f16m1_f16(a));
}

template <>
EIGEN_STRONG_INLINE PacketXh psqrt(const PacketXh& a) {
  return __riscv_vfsqrt_v_f16m1(a, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh print<PacketXh>(const PacketXh& a) {
  const PacketXh limit = pset1<PacketXh>(static_cast<Eigen::half>(1 << 10));
  const PacketXh abs_a = pabs(a);

  PacketMask16 mask = __riscv_vmfne_vv_f16m1_b16(a, a, unpacket_traits<PacketXh>::size);
  const PacketXh x = __riscv_vfadd_vv_f16m1_tumu(mask, a, a, a, unpacket_traits<PacketXh>::size);
  const PacketXh new_x = __riscv_vfcvt_f_x_v_f16m1(__riscv_vfcvt_x_f_v_i16m1(a, unpacket_traits<PacketXh>::size),
                                                   unpacket_traits<PacketXh>::size);

  mask = __riscv_vmflt_vv_f16m1_b16(abs_a, limit, unpacket_traits<PacketXh>::size);
  PacketXh signed_x = __riscv_vfsgnj_vv_f16m1(new_x, x, unpacket_traits<PacketXh>::size);
  return __riscv_vmerge_vvm_f16m1(x, signed_x, mask, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh pfloor<PacketXh>(const PacketXh& a) {
  PacketXh tmp = print<PacketXh>(a);
  // If greater, subtract one.
  PacketMask16 mask = __riscv_vmflt_vv_f16m1_b16(a, tmp, unpacket_traits<PacketXh>::size);
  return __riscv_vfsub_vf_f16m1_tumu(mask, tmp, tmp, static_cast<_Float16>(1.0), unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh preverse(const PacketXh& a) {
  PacketXsu idx = __riscv_vrsub_vx_u16m1(__riscv_vid_v_u16m1(unpacket_traits<PacketXh>::size),
                                         unpacket_traits<PacketXh>::size - 1, unpacket_traits<PacketXh>::size);
  return __riscv_vrgather_vv_f16m1(a, idx, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux<PacketXh>(const PacketXh& a) {
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredusum_vs_f16m1_f16m1(
      a, __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.0), unpacket_traits<PacketXh>::size),
      unpacket_traits<PacketXh>::size)));
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_mul<PacketXh>(const PacketXh& a) {
  // Multiply the vector by its reverse
  PacketXh prod = __riscv_vfmul_vv_f16m1(preverse(a), a, unpacket_traits<PacketXh>::size);
  PacketXh half_prod;

  if (EIGEN_RISCV64_RVV_VL >= 1024) {
    half_prod = __riscv_vslidedown_vx_f16m1(prod, 16, unpacket_traits<PacketXh>::size);
    prod = __riscv_vfmul_vv_f16m1(prod, half_prod, unpacket_traits<PacketXh>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 512) {
    half_prod = __riscv_vslidedown_vx_f16m1(prod, 8, unpacket_traits<PacketXh>::size);
    prod = __riscv_vfmul_vv_f16m1(prod, half_prod, unpacket_traits<PacketXh>::size);
  }
  if (EIGEN_RISCV64_RVV_VL >= 256) {
    half_prod = __riscv_vslidedown_vx_f16m1(prod, 4, unpacket_traits<PacketXh>::size);
    prod = __riscv_vfmul_vv_f16m1(prod, half_prod, unpacket_traits<PacketXh>::size);
  }
  // Last reduction
  half_prod = __riscv_vslidedown_vx_f16m1(prod, 2, unpacket_traits<PacketXh>::size);
  prod = __riscv_vfmul_vv_f16m1(prod, half_prod, unpacket_traits<PacketXh>::size);

  half_prod = __riscv_vslidedown_vx_f16m1(prod, 1, unpacket_traits<PacketXh>::size);
  prod = __riscv_vfmul_vv_f16m1(prod, half_prod, unpacket_traits<PacketXh>::size);

  // The reduction is done to the first element.
  return pfirst(prod);
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_min<PacketXh>(const PacketXh& a) {
  const Eigen::half max = (std::numeric_limits<Eigen::half>::max)();
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredmin_vs_f16m1_f16m1(
      a, __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(max), unpacket_traits<PacketXh>::size),
      unpacket_traits<PacketXh>::size)));
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_max<PacketXh>(const PacketXh& a) {
  const Eigen::half min = (std::numeric_limits<Eigen::half>::min)();
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredmax_vs_f16m1_f16m1(
      a, __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(min), unpacket_traits<PacketXh>::size),
      unpacket_traits<PacketXh>::size)));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXh, N>& kernel) {
  Eigen::half buffer[unpacket_traits<PacketXh>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(reinterpret_cast<_Float16*>(&buffer[i]), N * sizeof(Eigen::half), kernel.packet[i],
                   unpacket_traits<PacketXh>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] = __riscv_vle16_v_f16m1(reinterpret_cast<_Float16*>(&buffer[i * unpacket_traits<PacketXh>::size]),
                                             unpacket_traits<PacketXh>::size);
  }
}

EIGEN_STRONG_INLINE Packet2Xf half2float(const PacketXh& a) {
  return __riscv_vfwcvt_f_f_v_f32m2(a, unpacket_traits<Packet2Xf>::size);
}

EIGEN_STRONG_INLINE PacketXh float2half(const Packet2Xf& a) {
  return __riscv_vfncvt_f_f_w_f16m1(a, unpacket_traits<PacketXh>::size);
}

/********************************* Packet2Xh ************************************/

template <>
EIGEN_STRONG_INLINE Packet2Xh ptrue<Packet2Xh>(const Packet2Xh& /*a*/) {
  return __riscv_vreinterpret_f16m2(__riscv_vmv_v_x_u16m2(0xffffu, unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pzero<Packet2Xh>(const Packet2Xh& /*a*/) {
  return __riscv_vfmv_v_f_f16m2(static_cast<_Float16>(0.0), unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pabs(const Packet2Xh& a) {
  return __riscv_vfabs_v_f16m2(a, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pset1<Packet2Xh>(const Eigen::half& from) {
  return __riscv_vfmv_v_f_f16m2(numext::bit_cast<_Float16>(from), unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pset1frombits<Packet2Xh>(numext::uint16_t from) {
  return __riscv_vreinterpret_f16m2(__riscv_vmv_v_x_u16m2(from, unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh plset<Packet2Xh>(const Eigen::half& a) {
  Packet2Xh idx = __riscv_vfcvt_f_x_v_f16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(unpacket_traits<Packet4Xs>::size)),
      unpacket_traits<Packet2Xh>::size);
  return __riscv_vfadd_vf_f16m2(idx, numext::bit_cast<_Float16>(a), unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh padd<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfadd_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh psub<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfsub_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pnegate(const Packet2Xh& a) {
  return __riscv_vfneg_v_f16m2(a, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pconj(const Packet2Xh& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmul<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfmul_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pdiv<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfdiv_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmadd(const Packet2Xh& a, const Packet2Xh& b, const Packet2Xh& c) {
  return __riscv_vfmadd_vv_f16m2(a, b, c, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmsub(const Packet2Xh& a, const Packet2Xh& b, const Packet2Xh& c) {
  return __riscv_vfmsub_vv_f16m2(a, b, c, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pnmadd(const Packet2Xh& a, const Packet2Xh& b, const Packet2Xh& c) {
  return __riscv_vfnmsub_vv_f16m2(a, b, c, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pnmsub(const Packet2Xh& a, const Packet2Xh& b, const Packet2Xh& c) {
  return __riscv_vfnmadd_vv_f16m2(a, b, c, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmin<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  const Eigen::half nan = (std::numeric_limits<Eigen::half>::quiet_NaN)();
  Packet2Xh nans =
      __riscv_vfmv_v_f_f16m2(numext::bit_cast<_Float16>(nan), unpacket_traits<Packet2Xh>::size);
  PacketMask8 mask = __riscv_vmfeq_vv_f16m2_b8(a, a, unpacket_traits<Packet2Xh>::size);
  PacketMask8 mask2 = __riscv_vmfeq_vv_f16m2_b8(b, b, unpacket_traits<Packet2Xh>::size);
  mask = __riscv_vmand_mm_b8(mask, mask2, unpacket_traits<Packet2Xh>::size);

  return __riscv_vfmin_vv_f16m2_tumu(mask, nans, a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmin<PropagateNaN, Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return pmin<Packet2Xh>(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmin<PropagateNumbers, Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfmin_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmax<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  const Eigen::half nan = (std::numeric_limits<Eigen::half>::quiet_NaN)();
  Packet2Xh nans =
      __riscv_vfmv_v_f_f16m2(numext::bit_cast<_Float16>(nan), unpacket_traits<Packet2Xh>::size);
  PacketMask8 mask = __riscv_vmfeq_vv_f16m2_b8(a, a, unpacket_traits<Packet2Xh>::size);
  PacketMask8 mask2 = __riscv_vmfeq_vv_f16m2_b8(b, b, unpacket_traits<Packet2Xh>::size);
  mask = __riscv_vmand_mm_b8(mask, mask2, unpacket_traits<Packet2Xh>::size);

  return __riscv_vfmax_vv_f16m2_tumu(mask, nans, a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmax<PropagateNaN, Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return pmax<Packet2Xh>(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pmax<PropagateNumbers, Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vfmax_vv_f16m2(a, b, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcmp_le<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  PacketMask8 mask = __riscv_vmfle_vv_f16m2_b8(a, b, unpacket_traits<Packet2Xh>::size);
  return __riscv_vmerge_vvm_f16m2(pzero<Packet2Xh>(a), ptrue<Packet2Xh>(a), mask,
                                  unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcmp_lt<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  PacketMask8 mask = __riscv_vmflt_vv_f16m2_b8(a, b, unpacket_traits<Packet2Xh>::size);
  return __riscv_vmerge_vvm_f16m2(pzero<Packet2Xh>(a), ptrue<Packet2Xh>(a), mask,
                                  unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcmp_eq<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  PacketMask8 mask = __riscv_vmfeq_vv_f16m2_b8(a, b, unpacket_traits<Packet2Xh>::size);
  return __riscv_vmerge_vvm_f16m2(pzero<Packet2Xh>(a), ptrue<Packet2Xh>(a), mask,
                                  unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcmp_lt_or_nan<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  PacketMask8 mask = __riscv_vmfge_vv_f16m2_b8(a, b, unpacket_traits<Packet2Xh>::size);
  return __riscv_vfmerge_vfm_f16m2(ptrue<Packet2Xh>(a), static_cast<_Float16>(0.0), mask,
                                   unpacket_traits<Packet2Xh>::size);
}

// Logical Operations are not supported for half, so reinterpret casts
template <>
EIGEN_STRONG_INLINE Packet2Xh pand<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(a),
                                                                  __riscv_vreinterpret_v_f16m2_u16m2(b),
                                                                  unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh por<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(a),
                                                                 __riscv_vreinterpret_v_f16m2_u16m2(b),
                                                                 unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pxor<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vxor_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(a),
                                                                  __riscv_vreinterpret_v_f16m2_u16m2(b),
                                                                  unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pandnot<Packet2Xh>(const Packet2Xh& a, const Packet2Xh& b) {
  return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(
      __riscv_vreinterpret_v_f16m2_u16m2(a),
      __riscv_vnot_v_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(b), unpacket_traits<Packet2Xh>::size),
      unpacket_traits<Packet2Xh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pload<Packet2Xh>(const Eigen::half* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(from),
                                                        unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh ploadu<Packet2Xh>(const Eigen::half* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(from),
                                                          unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh ploaddup<Packet2Xh>(const Eigen::half* from) {
  Packet2Xsu idx = __riscv_vid_v_u16m2(unpacket_traits<Packet2Xh>::size);
  idx = __riscv_vand_vx_u16m2(idx, 0xfffeu, unpacket_traits<Packet2Xh>::size);
  return __riscv_vloxei16_v_f16m2(reinterpret_cast<const _Float16*>(from), idx, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh ploadquad<Packet2Xh>(const Eigen::half* from) {
  Packet2Xsu idx = __riscv_vid_v_u16m2(unpacket_traits<Packet2Xh>::size);
  idx = __riscv_vsrl_vx_u16m2(__riscv_vand_vx_u16m2(idx, 0xfffcu, unpacket_traits<Packet2Xh>::size), 1,
                              unpacket_traits<Packet2Xs>::size);
  return __riscv_vloxei16_v_f16m2(reinterpret_cast<const _Float16*>(from), idx, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet2Xh& from) {
  EIGEN_DEBUG_ALIGNED_STORE __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(to), from,
                                                  unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet2Xh& from) {
  EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(to), from,
                                                    unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_DEVICE_FUNC inline Packet2Xh pgather<Eigen::half, Packet2Xh>(const Eigen::half* from, Index stride) {
  return __riscv_vlse16_v_f16m2(reinterpret_cast<const _Float16*>(from), stride * sizeof(Eigen::half),
                                unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<Eigen::half, Packet2Xh>(Eigen::half* to, const Packet2Xh& from,
                                                                  Index stride) {
  __riscv_vsse16(reinterpret_cast<_Float16*>(to), stride * sizeof(Eigen::half), from,
                 unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Eigen::half pfirst<Packet2Xh>(const Packet2Xh& a) {
  return static_cast<Eigen::half>(__riscv_vfmv_f_s_f16m2_f16(a));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh psqrt(const Packet2Xh& a) {
  return __riscv_vfsqrt_v_f16m2(a, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh print<Packet2Xh>(const Packet2Xh& a) {
  const Packet2Xh limit = pset1<Packet2Xh>(static_cast<Eigen::half>(1 << 10));
  const Packet2Xh abs_a = pabs(a);

  PacketMask8 mask = __riscv_vmfne_vv_f16m2_b8(a, a, unpacket_traits<Packet2Xh>::size);
  const Packet2Xh x = __riscv_vfadd_vv_f16m2_tumu(mask, a, a, a, unpacket_traits<Packet2Xh>::size);
  const Packet2Xh new_x = __riscv_vfcvt_f_x_v_f16m2(
      __riscv_vfcvt_x_f_v_i16m2(a, unpacket_traits<Packet2Xh>::size), unpacket_traits<Packet2Xh>::size);

  mask = __riscv_vmflt_vv_f16m2_b8(abs_a, limit, unpacket_traits<Packet2Xh>::size);
  Packet2Xh signed_x = __riscv_vfsgnj_vv_f16m2(new_x, x, unpacket_traits<Packet2Xh>::size);
  return __riscv_vmerge_vvm_f16m2(x, signed_x, mask, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pfloor<Packet2Xh>(const Packet2Xh& a) {
  Packet2Xh tmp = print<Packet2Xh>(a);
  // If greater, subtract one.
  PacketMask8 mask = __riscv_vmflt_vv_f16m2_b8(a, tmp, unpacket_traits<Packet2Xh>::size);
  return __riscv_vfsub_vf_f16m2_tumu(mask, tmp, tmp, static_cast<_Float16>(1.0), unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh preverse(const Packet2Xh& a) {
  Packet2Xsu idx =
      __riscv_vrsub_vx_u16m2(__riscv_vid_v_u16m2(unpacket_traits<Packet2Xh>::size),
                             unpacket_traits<Packet2Xh>::size - 1, unpacket_traits<Packet2Xh>::size);
  return __riscv_vrgather_vv_f16m2(a, idx, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux<Packet2Xh>(const Packet2Xh& a) {
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredusum_vs_f16m2_f16m1(
      a, __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.0), unpacket_traits<Packet2Xh>::size / 2),
      unpacket_traits<Packet2Xh>::size)));
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_mul<Packet2Xh>(const Packet2Xh& a) {
  return predux_mul<PacketXh>(__riscv_vfmul_vv_f16m1(__riscv_vget_v_f16m2_f16m1(a, 0), __riscv_vget_v_f16m2_f16m1(a, 1),
                                                     unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_min<Packet2Xh>(const Packet2Xh& a) {
  const Eigen::half max = (std::numeric_limits<Eigen::half>::max)();
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredmin_vs_f16m2_f16m1(
      a, __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(max), unpacket_traits<Packet2Xh>::size / 2),
      unpacket_traits<Packet2Xh>::size)));
}

template <>
EIGEN_STRONG_INLINE Eigen::half predux_max<Packet2Xh>(const Packet2Xh& a) {
  const Eigen::half min = (std::numeric_limits<Eigen::half>::min)();
  return static_cast<Eigen::half>(__riscv_vfmv_f(__riscv_vfredmax_vs_f16m2_f16m1(
      a, __riscv_vfmv_v_f_f16m1(numext::bit_cast<_Float16>(min), unpacket_traits<Packet2Xh>::size / 2),
      unpacket_traits<Packet2Xh>::size)));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2Xh, N>& kernel) {
  Eigen::half buffer[unpacket_traits<Packet2Xh>::size * N];
  int i = 0;

  for (i = 0; i < N; i++) {
    __riscv_vsse16(reinterpret_cast<_Float16*>(&buffer[i]), N * sizeof(Eigen::half), kernel.packet[i],
                   unpacket_traits<Packet2Xh>::size);
  }

  for (i = 0; i < N; i++) {
    kernel.packet[i] =
        __riscv_vle16_v_f16m2(reinterpret_cast<_Float16*>(&buffer[i * unpacket_traits<Packet2Xh>::size]),
                              unpacket_traits<Packet2Xh>::size);
  }
}

EIGEN_STRONG_INLINE Packet4Xf half2float(const Packet2Xh& a) {
  return __riscv_vfwcvt_f_f_v_f32m4(a, unpacket_traits<Packet4Xf>::size);
}

EIGEN_STRONG_INLINE Packet2Xh float2half(const Packet4Xf& a) {
  return __riscv_vfncvt_f_f_w_f16m2(a, unpacket_traits<Packet2Xh>::size);
}

template <typename Packet = Packet2Xh>
EIGEN_STRONG_INLINE
typename std::enable_if<std::is_same<Packet, Packet2Xh>::value && (unpacket_traits<Packet2Xh>::size % 8) == 0,
                        PacketXh>::type
predux_half(const Packet2Xh& a) {
  return __riscv_vfadd_vv_f16m1(__riscv_vget_v_f16m2_f16m1(a, 0), __riscv_vget_v_f16m2_f16m1(a, 1),
                                unpacket_traits<PacketXh>::size);
}

F16_PACKET_FUNCTION(Packet2Xf, PacketXh, pcos)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, pexp)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, pexpm1)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, plog)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, plog1p)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, plog2)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, preciprocal)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, prsqrt)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, psin)
F16_PACKET_FUNCTION(Packet2Xf, PacketXh, ptanh)

F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, pcos)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, pexp)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, pexpm1)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, plog)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, plog1p)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, plog2)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, preciprocal)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, prsqrt)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, psin)
F16_PACKET_FUNCTION(Packet4Xf, Packet2Xh, ptanh)

/********************************* casting ************************************/

template <>
struct type_casting_traits<_Float16, numext::int16_t> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<numext::int16_t, _Float16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE PacketXh pcast<PacketXs, PacketXh>(const PacketXs& a) {
  return __riscv_vfcvt_f_x_v_f16m1(a, unpacket_traits<PacketXs>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXs pcast<PacketXh, PacketXs>(const PacketXh& a) {
  return __riscv_vfcvt_rtz_x_f_v_i16m1(a, unpacket_traits<PacketXh>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXh preinterpret<PacketXh, PacketXs>(const PacketXs& a) {
  return __riscv_vreinterpret_v_i16m1_f16m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketXs preinterpret<PacketXs, PacketXh>(const PacketXh& a) {
  return __riscv_vreinterpret_v_f16m1_i16m1(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcast<Packet2Xs, Packet2Xh>(const Packet2Xs& a) {
  return __riscv_vfcvt_f_x_v_f16m2(a, unpacket_traits<Packet2Xs>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xs pcast<Packet2Xh, Packet2Xs>(const Packet2Xh& a) {
  return __riscv_vfcvt_rtz_x_f_v_i16m2(a, unpacket_traits<Packet2Xh>::size);
}

template <>
EIGEN_STRONG_INLINE Packet2Xh preinterpret<Packet2Xh, Packet2Xs>(const Packet2Xs& a) {
  return __riscv_vreinterpret_v_i16m2_f16m2(a);
}

template <>
EIGEN_STRONG_INLINE Packet2Xs preinterpret<Packet2Xs, Packet2Xh>(const Packet2Xh& a) {
  return __riscv_vreinterpret_v_f16m2_i16m2(a);
}

template <>
EIGEN_STRONG_INLINE Packet4Xs pcast<PacketXh, Packet4Xs>(const PacketXh& a, const PacketXh& b, const PacketXh& c,
                                                               const PacketXh& d) {
  return __riscv_vcreate_v_i16m1_i16m4(__riscv_vfcvt_rtz_x_f_v_i16m1(a, unpacket_traits<PacketXh>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i16m1(b, unpacket_traits<PacketXh>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i16m1(c, unpacket_traits<PacketXh>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i16m1(d, unpacket_traits<PacketXh>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcast<PacketXs, Packet2Xh>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vcreate_v_f16m1_f16m2(__riscv_vfcvt_f_x_v_f16m1(a, unpacket_traits<PacketXs>::size),
                                       __riscv_vfcvt_f_x_v_f16m1(b, unpacket_traits<PacketXs>::size));
}

template <>
EIGEN_STRONG_INLINE Packet2Xh pcast<PacketXh, Packet2Xh>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vcreate_v_f16m1_f16m2(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet2Xs pcast<PacketXh, Packet2Xs>(const PacketXh& a, const PacketXh& b) {
  return __riscv_vcreate_v_i16m1_i16m2(__riscv_vfcvt_rtz_x_f_v_i16m1(a, unpacket_traits<PacketXh>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i16m1(b, unpacket_traits<PacketXh>::size));
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_FP16_RVV10_H
