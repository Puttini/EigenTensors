#ifndef TENSOR_TENSORMAP_H
#define TENSOR_TENSORMAP_H

#include <Eigen/Eigen>

template< typename Scalar, size_t rows = Eigen::Dynamic, size_t cols = Eigen::Dynamic >
using MatrixR = Eigen::Matrix< Scalar, rows, cols, Eigen::RowMajor>;

// ----------------------------------------------------------------------------------------

// These structures are used to help using big matrices of high dimensionality
// There is much to improve, but it is still useful for now
//
// It is more user friendly than Eigen's Tensor structes (when it works)
// Moreover, I had some troubles building Eigen's Tensor structes on Windows...

template< size_t dim >
struct Slice { size_t idx; };

typedef Eigen::InnerStride<Eigen::Dynamic> InnerStride;
typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;

template< typename Derived, typename Scalar, size_t dim >
struct TensorMapBase
{
    typedef Eigen::Ref< ConstAs< Scalar, MatrixR<NonConst<Scalar>> >,
            Eigen::Unaligned, Stride >
            EigenRef;

    static_assert( dim > 0, "You are a weird person" );

    Scalar* data_;
    size_t shape_[dim];
    size_t stride_[dim];

    template< typename ... Dimensions >
    TensorMapBase( Scalar* data, const InnerStride& inner_stride, Dimensions ... dimensions )
     : data_(data)
    {
        static_assert( sizeof...(dimensions) == dim, "You passed a wrong number of dimensions" );
        stride_[dim-1] = inner_stride.inner();
        _init<0>( dimensions... );
    }

    template< typename ... Dimensions >
    TensorMapBase( Scalar* data, Dimensions ... dimensions )
     : TensorMapBase( data, InnerStride(1), dimensions... )
    {}

    // Only takes the inner stride into account ! (for now)
    template< typename ... Dimensions >
    TensorMapBase( EigenRef&& mat, Dimensions ... dimensions )
     : TensorMapBase( mat.data(), InnerStride(mat.innerStride()), dimensions... )
    {
        assert( shape_[0] * stride_[0] == mat.rows()*mat.cols() );
    }

    // Used to initialize strides and shapes in constructors
    template< size_t s, typename ... OtherDimensions >
    inline void _init( size_t d, OtherDimensions ... other_dimensions )
    {
        shape_[s] = d;
        _init<s+1>( other_dimensions... );
        stride_[s] = shape_[s+1] * stride_[s+1];
    }
    template< size_t s >
    inline void _init( size_t d )
    {
        shape_[s] = d;
    }
};

#define TENSOR_MAP_BASE( Derived, Scalar, dim ) TensorMapBase< \
    ConstAs< Scalar, Derived< NonConst<Scalar>, dim > >, \
    Scalar, dim >

template< typename Scalar, size_t dim >
struct TensorMap : public TENSOR_MAP_BASE( TensorMap, Scalar, dim )
{
    typedef TENSOR_MAP_BASE( TensorMap, Scalar, dim ) Base;
    using Base::Base;
};

#endif //TENSOR_TENSORMAP_H
