#ifndef FPM_FIXED_HPP
#define FPM_FIXED_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <exception>

namespace fpm
{
    
	template <size_t I, size_t F, bool S, bool R>
	class fixed;

	namespace detail
	{
		// helper templates to make magic with types :)
		// these allow us to determine reasonable types from
		// a desired size, they also let us infer the next largest type
		// from a type which is nice for the division op
		template <size_t T>
		struct type_from_size
		{
			using value_type = void;
			using unsigned_type = void;
			using signed_type = void;
			static constexpr bool is_specialized = false;
		};

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__STRICT_ANSI__)
		template <>
		struct type_from_size<128>
		{
			static constexpr bool is_specialized = true;
			static constexpr size_t size = 128;

			using value_type = __int128;
			using unsigned_type = unsigned __int128;
			using signed_type = __int128;
			using next_size = type_from_size<256>;
		};
#endif

		template <>
		struct type_from_size<64>
		{
			static constexpr bool is_specialized = true;
			static constexpr size_t size = 64;

			using value_type = int64_t;
			using unsigned_type = ::std::make_unsigned<value_type>::type;
			using signed_type = ::std::make_signed<value_type>::type;
			using next_size = type_from_size<128>;
		};

		template <>
		struct type_from_size<32>
		{
			static constexpr bool is_specialized = true;
			static constexpr size_t size = 32;

			using value_type = int32_t;
			using unsigned_type = ::std::make_unsigned<value_type>::type;
			using signed_type = ::std::make_signed<value_type>::type;
			using next_size = type_from_size<64>;
		};

		template <>
		struct type_from_size<16>
		{
			static constexpr bool is_specialized = true;
			static constexpr size_t size = 16;

			using value_type = int16_t;
			using unsigned_type = ::std::make_unsigned<value_type>::type;
			using signed_type = ::std::make_signed<value_type>::type;
			using next_size = type_from_size<32>;
		};

		template <>
		struct type_from_size<8>
		{
			static constexpr bool is_specialized = true;
			static constexpr size_t size = 8;

			using value_type = int8_t;
			using unsigned_type = ::std::make_unsigned<value_type>::type;
			using signed_type = ::std::make_signed<value_type>::type;
			using next_size = type_from_size<16>;
		};

		// this is to assist in adding support for non-native base
		// types (for adding big-int support), this should be fine
		// unless your bit-int class doesn't nicely support casting
		template <class B, class N>
		constexpr B next_to_base(N rhs)
		{
			return static_cast<B>(rhs);
		}

		struct divide_by_zero : ::std::exception
		{
		};

		// template <size_t I, size_t F, bool S, bool R>
		// constexpr fixed<I, F, S, R> divide(fixed<I, F, S, R> numerator, fixed<I, F, S, R> denominator, fixed<I, F, S, R> &remainder, typename ::std::enable_if<type_from_size<I + F>::next_size::is_specialized>::type * = nullptr)
		// {

		// 	using next_type = typename fixed<I, F>::next_type;
		// 	using base_type = typename fixed<I, F>::base_type;
		// 	constexpr size_t fractional_bits = fixed<I, F>::fractional_bits;

		// 	next_type t(numerator.to_raw());
		// 	t <<= fractional_bits;

		// 	fixed<I, F, S, R> quotient;

		// 	quotient = fixed<I, F>::from_base(next_to_base<base_type>(t / denominator.to_raw()));
		// 	remainder = fixed<I, F>::from_base(next_to_base<base_type>(t % denominator.to_raw()));

		// 	return quotient;
		// }

		// template <size_t I, size_t F, bool S, bool R>
		// constexpr fixed<I, F, S, R> divide(fixed<I, F, S, R> numerator, fixed<I, F, S, R> denominator, fixed<I, F, S, R> &remainder, typename ::std::enable_if<!type_from_size<I + F>::next_size::is_specialized>::type * = nullptr)
		// {

		// 	using unsigned_type = typename fixed<I, F>::unsigned_type;

		// 	constexpr int bits = fixed<I, F>::total_bits;

		// 	if (denominator == 0)
		// 	{
		// 		throw divide_by_zero();
		// 	}
		// 	else
		// 	{

		// 		int sign = 0;

		// 		fixed<I, F, S, R> quotient;

		// 		if (numerator < 0)
		// 		{
		// 			sign ^= 1;
		// 			numerator = -numerator;
		// 		}

		// 		if (denominator < 0)
		// 		{
		// 			sign ^= 1;
		// 			denominator = -denominator;
		// 		}

		// 		unsigned_type n = numerator.to_raw();
		// 		unsigned_type d = denominator.to_raw();
		// 		unsigned_type x = 1;
		// 		unsigned_type answer = 0;

		// 		// egyptian division algorithm
		// 		while ((n >= d) && (((d >> (bits - 1)) & 1) == 0))
		// 		{
		// 			x <<= 1;
		// 			d <<= 1;
		// 		}

		// 		while (x != 0)
		// 		{
		// 			if (n >= d)
		// 			{
		// 				n -= d;
		// 				answer += x;
		// 			}

		// 			x >>= 1;
		// 			d >>= 1;
		// 		}

		// 		unsigned_type l1 = n;
		// 		unsigned_type l2 = denominator.to_raw();

		// 		// calculate the lower bits (needs to be unsigned)
		// 		while (l1 >> (bits - F) > 0)
		// 		{
		// 			l1 >>= 1;
		// 			l2 >>= 1;
		// 		}
		// 		const unsigned_type lo = (l1 << F) / l2;

		// 		quotient = fixed<I, F>::from_base((answer << F) | lo);
		// 		remainder = n;

		// 		if (sign)
		// 		{
		// 			quotient = -quotient;
		// 		}

		// 		return quotient;
		// 	}
		// }

		// // this is the usual implementation of multiplication
		// template <size_t I, size_t F, bool S, bool R>
		// constexpr fixed<I, F, S, R> multiply(fixed<I, F, S, R> lhs, fixed<I, F, S, R> rhs, typename ::std::enable_if<type_from_size<I + F>::next_size::is_specialized>::type * = nullptr)
		// {

		// 	using next_type = typename fixed<I, F>::next_type;
		// 	using base_type = typename fixed<I, F>::base_type;

		// 	constexpr size_t fractional_bits = fixed<I, F>::fractional_bits;

		// 	next_type t(static_cast<next_type>(lhs.to_raw()) * static_cast<next_type>(rhs.to_raw()));
		// 	t >>= fractional_bits;

		// 	return fixed<I, F>::from_base(next_to_base<base_type>(t));
		// }

		// // this is the fall back version we use when we don't have a next size
		// // it is slightly slower, but is more robust since it doesn't
		// // require and upgraded type
		// template <size_t I, size_t F, bool S, bool R>
		// constexpr fixed<I, F, S, R> multiply(fixed<I, F, S, R> lhs, fixed<I, F, S, R> rhs, typename ::std::enable_if<!type_from_size<I + F>::next_size::is_specialized>::type * = nullptr)
		// {

		// 	using base_type = typename fixed<I, F>::base_type;

		// 	constexpr size_t fractional_bits = fixed<I, F>::fractional_bits;
		// 	constexpr base_type integer_mask = fixed<I, F>::integer_mask;
		// 	constexpr base_type fractional_mask = fixed<I, F>::fractional_mask;

		// 	// more costly but doesn't need a larger type
		// 	const base_type a_hi = (lhs.to_raw() & integer_mask) >> fractional_bits;
		// 	const base_type b_hi = (rhs.to_raw() & integer_mask) >> fractional_bits;
		// 	const base_type a_lo = (lhs.to_raw() & fractional_mask);
		// 	const base_type b_lo = (rhs.to_raw() & fractional_mask);

		// 	const base_type x1 = a_hi * b_hi;
		// 	const base_type x2 = a_hi * b_lo;
		// 	const base_type x3 = a_lo * b_hi;
		// 	const base_type x4 = a_lo * b_lo;

		// 	return fixed<I, F>::from_base((x1 << fractional_bits) + (x3 + x2) + (x4 >> fractional_bits));
		// }
	}

//! Fixed-point number type
//! \tparam BaseType         the base integer type used to store the fixed-point number. This can be a signed or unsigned type.
//! \tparam IntermediateType the integer type used to store intermediate results during calculations.
//! \tparam FractionBits     the number of bits of the BaseType used to store the fraction
//! \tparam EnableRounding   enable rounding of LSB for multiplication, division, and type conversion
// template <typename BaseType, typename IntermediateType, size_t FractionBits, bool EnableRounding = true>
// these types are identified from the integral and fractional bit numbers

template <size_t IntegralBits, size_t FractionBits, bool Signed = true, bool EnableRounding = false>
class fixed
{
public:
    using BaseType = typename std::conditional<Signed, typename detail::type_from_size<IntegralBits + FractionBits>::signed_type, typename detail::type_from_size<IntegralBits + FractionBits>::unsigned_type>::type;
    using IntermediateType = typename std::conditional<Signed, typename detail::type_from_size<IntegralBits + FractionBits>::next_size::signed_type, typename detail::type_from_size<IntegralBits + FractionBits>::next_size::unsigned_type>::type;

private:
    static_assert(::std::is_integral<BaseType>::value, "BaseType must be an integral type");
    static_assert(FractionBits > 0, "FractionBits must be greater than zero");
    static_assert(FractionBits <= sizeof(BaseType) * 8 - 1, "BaseType must at least be able to contain entire fraction, with space for at least one integral bit");

    //the following assert won't allow for intermediate types that are not specialized, so no need to use detail::multiply and detail::divide
    static_assert(sizeof(IntermediateType) > sizeof(BaseType), "IntermediateType must be larger than BaseType");
    static_assert(std::is_signed<IntermediateType>::value == std::is_signed<BaseType>::value, "IntermediateType must have same signedness as BaseType");

    // Although this value fits in the BaseType in terms of bits, if there's only one integral bit, this value
    // is incorrect (flips from positive to negative), so we must extend the size to IntermediateType.
    using FractionMultType = typename std::conditional<(sizeof(BaseType) * 8 - FractionBits == 1),IntermediateType,BaseType>::type;

    static constexpr FractionMultType FRACTION_MULT = FractionMultType(1) << FractionBits;

    struct raw_construct_tag {};
    constexpr inline fixed(BaseType val, raw_construct_tag) noexcept : m_value(val) {}

public:
    inline fixed() noexcept = default;
    typedef fixed<IntegralBits, FractionBits, false, EnableRounding> UnsignedType;
    typedef fixed<IntegralBits, FractionBits, true, EnableRounding> SignedType;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverflow"
#endif
	static constexpr BaseType FractionalMask = ~(static_cast<typename UnsignedType::BaseType>(~BaseType(0)) << FractionBits);
	static constexpr BaseType IntegerMask    = ~FractionalMask;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // Converts an integral number to the fixed-point type.
    // Like static_cast, this truncates bits that don't fit.
    template <typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
    constexpr inline explicit fixed(T val) noexcept
        : m_value(static_cast<BaseType>(val * FRACTION_MULT))
    {}

    // Converts an floating-point number to the fixed-point type.
    // Like static_cast, this truncates bits that don't fit.
    template <typename T, typename ::std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    constexpr inline explicit fixed(T val) noexcept
        : m_value(static_cast<BaseType>((EnableRounding) ?
		       (val >= 0.0) ? (val * FRACTION_MULT + T{0.5}) : (val * FRACTION_MULT - T{0.5})
		      : (val * FRACTION_MULT)))
    {}

    // Constructs from another fixed-point type with possibly different underlying representation.
    // Like static_cast, this truncates bits that don't fit.
    template <size_t I, size_t F, bool S, bool R>
    constexpr inline explicit fixed(fixed<I, F, S, R> val) noexcept
        : m_value(from_fixed_point<F>(val.raw_value()).raw_value())
    {}

    // Explicit conversion to a floating-point type
    template <typename T, typename ::std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    constexpr inline explicit operator T() const noexcept
    {
        return static_cast<T>(m_value) / FRACTION_MULT;
    }

    // Explicit conversion to an integral type
    template <typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
    constexpr inline explicit operator T() const noexcept
    {
        return static_cast<T>(m_value / FRACTION_MULT);
    }

    template <typename T>
    constexpr inline fixed& operator=(T val) noexcept
    {
        m_value = fixed(val).raw_value();
        return *this;
    } 

    template <typename T>
    constexpr inline bool operator < (T val) const noexcept
    {
        return m_value < fixed(val).raw_value();
    }

    template <typename T>
    constexpr inline bool operator <= (T val) const noexcept
    {
        return m_value <= fixed(val).raw_value();
    }

    template <typename T>
    constexpr inline bool operator > (T val) const noexcept
    {
        return m_value > fixed(val).raw_value();
    }

    template <typename T>
    constexpr inline bool operator >= (T val) const noexcept
    {
        return m_value >= fixed(val).raw_value();
    }
    
    // Returns the raw underlying value of this type.
    // Do not use this unless you know what you're doing.
    constexpr inline BaseType raw_value() const noexcept
    {
        return m_value;
    }

    // Returns the fractional part of the fixed-point number.
    constexpr inline BaseType fraction() const noexcept
    {
        return m_value & (FRACTION_MULT - 1);
    }

    constexpr inline fixed mod1() const noexcept
    {
        return fixed::from_raw_value(m_value & (FRACTION_MULT - 1));
    }

    // Returns the integral part of the fixed-point number.
    constexpr inline BaseType integral() const noexcept
    {
        return m_value / FRACTION_MULT;
    }

    // Reinterprets the fixed-point number as unsigned.
    constexpr inline UnsignedType to_unsigned() const noexcept
    {
        return UnsignedType::from_raw_value(m_value);
    }

    // Reinterprets the fixed-point number as signed.
    constexpr inline SignedType to_signed() const noexcept
    {
        return SignedType::from_raw_value(m_value);
    }

    // Returns the rounded value of the fixed-point number.
    constexpr inline fixed round() const noexcept
    {
        return fixed((m_value >= 0) ? (m_value + FRACTION_MULT / 2) : (m_value - FRACTION_MULT / 2), raw_construct_tag{});
    }

    // bitwise operators
    constexpr inline fixed operator~() const noexcept
    {
        return fixed(~m_value, raw_construct_tag{});
    }

    constexpr inline fixed operator&(const fixed& y) const noexcept
    {
        return fixed(m_value & y.m_value, raw_construct_tag{});
    }

    constexpr inline fixed operator|(const fixed& y) const noexcept
    {
        return fixed(m_value | y.m_value, raw_construct_tag{});
    }

    constexpr inline fixed operator^(const fixed& y) const noexcept
    {
        return fixed(m_value ^ y.m_value, raw_construct_tag{});
    }

    constexpr inline fixed operator<<(int shift) const noexcept
    {
        return fixed(m_value << shift, raw_construct_tag{});
    }

    constexpr inline fixed operator>>(int shift) const noexcept
    {
        return fixed(m_value >> shift, raw_construct_tag{});
    }

    // bitwise assignment operators
    inline fixed& operator&=(const fixed& y) noexcept
    {
        m_value &= y.m_value;
        return *this;
    }

    inline fixed& operator|=(const fixed& y) noexcept
    {
        m_value |= y.m_value;
        return *this;
    }

    inline fixed& operator^=(const fixed& y) noexcept
    {
        m_value ^= y.m_value;
        return *this;
    }

    inline fixed& operator<<=(int shift) noexcept
    {
        m_value <<= shift;
        return *this;
    }

    inline fixed& operator>>=(int shift) noexcept
    {
        m_value >>= shift;
        return *this;
    }

    //! Constructs a fixed-point number from another fixed-point number.
    //! \tparam NumFractionBits the number of bits used by the fraction in \a value.
    //! \param value the integer fixed-point number
    template <size_t NumFractionBits, typename T, typename ::std::enable_if<(NumFractionBits > FractionBits)>::type* = nullptr>
    static constexpr inline fixed from_fixed_point(T value) noexcept
    {
	// To correctly round the last bit in the result, we need one more bit of information.
	// We do this by multiplying by two before dividing and adding the LSB to the real result.
	return (EnableRounding) ? fixed(static_cast<BaseType>(
             value / (T(1) << (NumFractionBits - FractionBits)) +
            (value / (T(1) << (NumFractionBits - FractionBits - 1)) % 2)),
	    raw_construct_tag{}) :
	    fixed(static_cast<BaseType>(value / (T(1) << (NumFractionBits - FractionBits))),
	     raw_construct_tag{});
    }

    template <size_t NumFractionBits, typename T, typename ::std::enable_if<(NumFractionBits <= FractionBits)>::type* = nullptr>
    static constexpr inline fixed from_fixed_point(T value) noexcept
    {
        return fixed(static_cast<BaseType>(
            value * (T(1) << (FractionBits - NumFractionBits))),
            raw_construct_tag{});
    }

    // Constructs a fixed-point number from its raw underlying value.
    // Do not use this unless you know what you're doing.
    static constexpr inline fixed from_raw_value(BaseType value) noexcept
    {
        return fixed(value, raw_construct_tag{});
    }

    //
    // Constants
    //
    static constexpr fixed e() { return from_fixed_point<61>(6267931151224907085ll); }
    static constexpr fixed pi() { return from_fixed_point<61>(7244019458077122842ll); }
    static constexpr fixed half_pi() { return from_fixed_point<62>(7244019458077122842ll); }
    static constexpr fixed two_pi() { return from_fixed_point<60>(7244019458077122842ll); }

    //
    // Arithmetic member operators
    //

    constexpr inline fixed operator-() const noexcept
    {
        return fixed::from_raw_value(-m_value);
    }

    inline fixed& operator+=(const fixed& y) noexcept
    {
        m_value += y.m_value;
        return *this;
    }

    template <typename I, typename ::std::enable_if<::std::is_integral<I>::value>::type* = nullptr>
    inline fixed& operator+=(I y) noexcept
    {
        m_value += y * FRACTION_MULT;
        return *this;
    }

    inline fixed& operator-=(const fixed& y) noexcept
    {
        m_value -= y.m_value;
        return *this;
    }

    template <typename I, typename ::std::enable_if<::std::is_integral<I>::value>::type* = nullptr>
    inline fixed& operator-=(I y) noexcept
    {
        m_value -= y * FRACTION_MULT;
        return *this;
    }

    //could be updated to use detail::multiply and remove intermediate type from the class
    inline fixed& operator*=(const fixed& y) noexcept
    {
	if (EnableRounding){
	    // Normal fixed-point multiplication is: x * y / 2**FractionBits.
	    // To correctly round the last bit in the result, we need one more bit of information.
	    // We do this by multiplying by two before dividing and adding the LSB to the real result.
	    auto value = (static_cast<IntermediateType>(m_value) * y.m_value) / (FRACTION_MULT / 2);
	    m_value = static_cast<BaseType>((value / 2) + (value % 2));
	} else {
	    auto value = (static_cast<IntermediateType>(m_value) * y.m_value) / FRACTION_MULT;
	    m_value = static_cast<BaseType>(value);
	}
	return *this;
    }

    template <typename I, typename ::std::enable_if<::std::is_integral<I>::value>::type* = nullptr>
    inline fixed& operator*=(I y) noexcept
    {
        m_value *= y;
        return *this;
    }

    //could be updated to use detail::divide and remove intermediate type from the class
    inline fixed& operator/=(const fixed& y) noexcept
    {
        assert(y.m_value != 0);
	if (EnableRounding){
	    // Normal fixed-point division is: x * 2**FractionBits / y.
	    // To correctly round the last bit in the result, we need one more bit of information.
	    // We do this by multiplying by two before dividing and adding the LSB to the real result.
	    auto value = (static_cast<IntermediateType>(m_value) * FRACTION_MULT * 2) / y.m_value;
	    m_value = static_cast<BaseType>((value / 2) + (value % 2));
	} else {
	    auto value = (static_cast<IntermediateType>(m_value) * FRACTION_MULT) / y.m_value;
	    m_value = static_cast<BaseType>(value);
	}
        return *this;
    }

    template <typename I, typename ::std::enable_if<::std::is_integral<I>::value>::type* = nullptr>
    inline fixed& operator/=(I y) noexcept
    {
        m_value /= y;
        return *this;
    }

private:
    BaseType m_value;
};

//
// Convenience typedefs
//

using fixed_16_16 = fixed<16, 16>;
using fixed_24_8 = fixed<24, 8>;
using fixed_8_24 = fixed<8, 24>;

//
// Addition
//

template <size_t I, size_t F, bool S, bool R>
constexpr inline fixed<I, F, S, R> operator+(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) += y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator+(const fixed<I, F, S, R>& x, T y) noexcept
{
    return fixed<I, F, S, R>(x) += y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator+(T x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(y) += x;
}

//
// Subtraction
//

template <size_t I, size_t F, bool S, bool R>
constexpr inline fixed<I, F, S, R> operator-(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) -= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator-(const fixed<I, F, S, R>& x, T y) noexcept
{
    return fixed<I, F, S, R>(x) -= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator-(T x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) -= y;
}

//
// Multiplication
//

template <size_t I, size_t F, bool S, bool R>
constexpr inline fixed<I, F, S, R> operator*(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) *= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator*(const fixed<I, F, S, R>& x, T y) noexcept
{
    return fixed<I, F, S, R>(x) *= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator*(T x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(y) *= x;
}

// template <size_t I, size_t F, bool S, bool R>
// constexpr inline fixed<I, F, S, R> operator*(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
// {
//     fixed<I, F, S, R> result = x;
//     result *= y;
//     return result;
// }

//
// Division
//

template <size_t I, size_t F, bool S, bool R>
constexpr inline fixed<I, F, S, R> operator/(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) /= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator/(const fixed<I, F, S, R>& x, T y) noexcept
{
    return fixed<I, F, S, R>(x) /= y;
}

template <size_t I, size_t F, bool S, bool R, typename T, typename ::std::enable_if<::std::is_integral<T>::value>::type* = nullptr>
constexpr inline fixed<I, F, S, R> operator/(T x, const fixed<I, F, S, R>& y) noexcept
{
    return fixed<I, F, S, R>(x) /= y;
}

//
// Comparison operators
//

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator==(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() == y.raw_value();
}

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator!=(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() != y.raw_value();
}

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator<(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() < y.raw_value();
}

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator>(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() > y.raw_value();
}

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator<=(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() <= y.raw_value();
}

template <size_t I, size_t F, bool S, bool R>
constexpr inline bool operator>=(const fixed<I, F, S, R>& x, const fixed<I, F, S, R>& y) noexcept
{
    return x.raw_value() >= y.raw_value();
}

namespace detail
{
// Number of base-10 digits required to fully represent a number of bits
static constexpr int max_digits10(int bits)
{
    // 8.24 fixed-point equivalent of (int)ceil(bits * std::log10(2));
    using T = long long;
    return static_cast<int>((T{bits} * 5050445 + (T{1} << 24) - 1) >> 24);
}

// Number of base-10 digits that can be fully represented by a number of bits
static constexpr int digits10(int bits)
{
    // 8.24 fixed-point equivalent of (int)(bits * std::log10(2));
    using T = long long;
    return static_cast<int>((T{bits} * 5050445) >> 24);
}

} // namespace detail
} // namespace fpm

// Specializations for customization points
namespace std
{

template <size_t I, size_t F, bool S, bool R>
struct hash<fpm::fixed<I, F, S, R>>
{
    using argument_type = fpm::fixed<I, F, S, R>;
    using result_type = std::size_t;

    result_type operator()(argument_type arg) const noexcept(noexcept(std::declval<std::hash<typename argument_type::BaseType>>()(arg.raw_value()))) {
        return m_hash(arg.raw_value());
    }

private:
    std::hash<typename argument_type::BaseType> m_hash;
};

template <size_t I, size_t F, bool S, bool R>
struct numeric_limits<fpm::fixed<I,F,S,R>>
{
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::is_signed;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = true;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::is_modulo;
    static constexpr int digits = std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::digits;

    // Any number with `digits10` significant base-10 digits (that fits in
    // the range of the type) is guaranteed to be convertible from text and
    // back without change. Worst case, this is 0.000...001, so we can only
    // guarantee this case. Nothing more.
    static constexpr int digits10 = 1;

    // This is equal to max_digits10 for the integer and fractional part together.
    static constexpr int max_digits10 =
        fpm::detail::max_digits10(std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::digits - F) + fpm::detail::max_digits10(F);

    static constexpr int radix = 2;
    static constexpr int min_exponent = 1 - F;
    static constexpr int min_exponent10 = -fpm::detail::digits10(F);
    static constexpr int max_exponent = std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::digits - F;
    static constexpr int max_exponent10 = fpm::detail::digits10(std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::digits - F);
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = false;

    static constexpr fpm::fixed<I, F, S, R> lowest() noexcept {
        return fpm::fixed<I, F, S, R>::from_raw_value(std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::lowest());
    };

    static constexpr fpm::fixed<I, F, S, R> min() noexcept {
        return lowest();
    }

    static constexpr fpm::fixed<I, F, S, R> max() noexcept {
        return fpm::fixed<I, F, S, R>::from_raw_value(std::numeric_limits<typename fpm::fixed<I, F, S, R>::BaseType>::max());
    };

    static constexpr fpm::fixed<I, F, S, R> epsilon() noexcept {
        return fpm::fixed<I, F, S, R>::from_raw_value(1);
    };

    static constexpr fpm::fixed<I, F, S, R> round_error() noexcept {
        return fpm::fixed<I, F, S, R>(1) / 2;
    };

    static constexpr fpm::fixed<I, F, S, R> denorm_min() noexcept {
        return min();
    }
};

template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_specialized;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_signed;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_integer;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_exact;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::has_infinity;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::has_quiet_NaN;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::has_signaling_NaN;
template <size_t I, size_t F, bool S, bool R>
constexpr std::float_denorm_style numeric_limits<fpm::fixed<I, F, S, R>>::has_denorm;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::has_denorm_loss;
template <size_t I, size_t F, bool S, bool R>
constexpr std::float_round_style numeric_limits<fpm::fixed<I, F, S, R>>::round_style;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_iec559;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_bounded;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::is_modulo;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::digits;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::digits10;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::max_digits10;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::radix;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::min_exponent;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::min_exponent10;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::max_exponent;
template <size_t I, size_t F, bool S, bool R>
constexpr int numeric_limits<fpm::fixed<I, F, S, R>>::max_exponent10;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::traps;
template <size_t I, size_t F, bool S, bool R>
constexpr bool numeric_limits<fpm::fixed<I, F, S, R>>::tinyness_before;

}

namespace fpm
{
template<typename T>
struct is_fixed : std::false_type {};

template<size_t IntegralBits, size_t FractionBits, bool Signed, bool EnableRounding>
struct is_fixed<fixed<IntegralBits, FractionBits, Signed, EnableRounding>> : std::true_type {};

#if  __cplusplus >= 201703L
template<typename T>
inline constexpr bool is_fixed_v = is_fixed<T>::value;
#endif
}
#endif
