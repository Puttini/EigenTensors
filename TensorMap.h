#ifndef TENSOR_TENSORMAP_H
#define TENSOR_TENSORMAP_H

#include <Eigen/Eigen>

template< typename Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixR = Eigen::Matrix< Scalar, rows, cols, Eigen::RowMajor>;

// ----------------------------------------------------------------------------------------

// These structures are used to help using big matrices of high dimensionality
// There is much to improve, but it is still useful for now
//
// It is more user friendly than Eigen's Tensor structes (when it works)
// Moreover, I had some troubles building Eigen's Tensor structes on Windows...

template< size_t dim >
struct Slice
{
    size_t idx;

    Slice( size_t idx = 0 )
     : idx(idx)
    {}
};

typedef Eigen::InnerStride<Eigen::Dynamic> InnerStride;
typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;

template< typename Derived, typename ConstDerived, typename SubDerived, typename SuperDerived, typename Scalar, size_t dim >
struct TensorMapBase
{
    typedef Eigen::Ref< ConstAs< Scalar, MatrixR<NonConst<Scalar>> >,
            Eigen::Unaligned, Stride >
            EigenRef;

    static_assert( dim > 0, "You are a weird person" );

    Scalar* data_;
    size_t shape_[dim];
    size_t stride_[dim];

    Derived& derived() { return *static_cast<Derived*>(this); }
    const ConstDerived& derived() const { return *static_cast<const ConstDerived*>(this); }

    // For debug !!
    TensorMapBase()
    {}

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
        assert( shape_[0]*stride_[0] == mat.rows()*mat.cols() && "Dimensions do not match the Eigen matrix" );
    }

    TensorMapBase( Derived& other )
    {
        data_ = other.data_;
        std::copy( other.stride_, other.stride_+dim, stride_ );
        std::copy( other.shape_, other.shape_+dim, shape_ );
    }
    TensorMapBase( NonConst<Derived>&& other )
    {
        data_ = other.data_;
        std::copy( other.stride_, other.stride_+dim, stride_ );
        std::copy( other.shape_, other.shape_+dim, shape_ );
    }

    // Slicing construction
    template< size_t SliceDim >
    TensorMapBase( const Slice<SliceDim>& slice, SuperDerived& super )
    {
        static_assert( SliceDim < dim, "Slice used on invalid dimension" );
        assert( slice.idx < super.shape_[SliceDim] && "Index out of shape" );
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy( super.stride_, super.stride_+SliceDim, stride_ );
        std::copy( super.shape_, super.shape_+SliceDim, shape_ );
        std::copy( super.stride_+SliceDim+1, super.stride_+dim+1, stride_+SliceDim );
        std::copy( super.shape_+SliceDim+1, super.shape_+dim+1, shape_+SliceDim );
    }
    template< size_t SliceDim >
    TensorMapBase( const Slice<SliceDim>& slice, SuperDerived&& super )
    {
        static_assert( SliceDim < dim, "Slice used on invalid dimension" );
        assert( slice.idx < super.shape_[SliceDim] && "Index out of shape" );
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy( super.stride_, super.stride_+SliceDim, stride_ );
        std::copy( super.shape_, super.shape_+SliceDim, shape_ );
        std::copy( super.stride_+SliceDim+1, super.stride_+dim+1, stride_+SliceDim );
        std::copy( super.shape_+SliceDim+1, super.shape_+dim+1, shape_+SliceDim );
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

    template< size_t SliceDim >
    SubDerived slice( size_t idx ) const
    {
        return SubDerived( Slice<SliceDim>(idx), derived() );
    }
    template< size_t SliceDim >
    SubDerived slice( size_t idx )
    { return SubDerived( Slice<SliceDim>(idx), derived() ); }
};

#define TENSOR_MAP_BASE( Derived, Scalar, dim, current_dim ) TensorMapBase< \
    ConstAs< Scalar, Derived< NonConst<Scalar>, dim, current_dim > >, \
    Const< Derived< Const<Scalar>, dim, current_dim > >, \
    Derived< Scalar, dim-1, current_dim >, \
    ConstAs< Scalar, Derived< Scalar, dim+1, current_dim > >, \
    Scalar, dim >

template< typename Scalar, size_t dim, size_t current_dim = 0 >
struct TensorMap : public TENSOR_MAP_BASE( TensorMap, Scalar, dim, current_dim )
{
    typedef TENSOR_MAP_BASE( TensorMap, Scalar, dim, current_dim ) Base;

    using Base::Base;
};

#endif //TENSOR_TENSORMAP_H
