#ifndef TENSOR_TENSORMAP_H
#define TENSOR_TENSORMAP_H

#include <Eigen/Eigen>

template< typename Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixR = Eigen::Matrix< Scalar, rows, cols, Eigen::RowMajor >;
template< typename Scalar, int size = Eigen::Dynamic >
using Vector = Eigen::Matrix< Scalar, size, 1, Eigen::ColMajor >;

// ----------------------------------------------------------------------------------------

// These structures are used to help using big matrices of high dimensionality
// There is much to improve, but it is still useful for now
//
// It is more user friendly than Eigen's Tensor structes (when it works)
// Moreover, I had some troubles building Eigen's Tensor structes on Windows...

// ----------------------------------------------------------------------------------------

template< typename Scalar, size_t dim, size_t current_dim = 0 >
class TensorMap;

// ----------------------------------------------------------------------------------------

namespace TensorMapTools
{

// These little structs are used to keep easy-to-read constructors
// of TensorMapBase

template<size_t dim>
struct Slice
{
    size_t idx;

    Slice(size_t idx = 0)
            : idx(idx) {}
};

template< typename Integer >
struct Shape
{
    const Integer* shape;

    Shape( const Integer* shape )
     : shape(shape)
    {}
};

template< typename Integer >
struct Stride
{
    const Integer* stride;

    Stride( const Integer* stride )
     : stride(stride)
    {}
};

// Dynamic strides
typedef Eigen::InnerStride<Eigen::Dynamic> DynInnerStride;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

// ----------------------------------------------------------------------------------------

template<typename Derived, typename ConstDerived, typename SubDerived, typename SuperDerived, typename Scalar, size_t dim>
class TensorMapBase
{
    typedef Eigen::Ref<ConstAs<Scalar, MatrixR<NonConst<Scalar>>>,
            Eigen::Unaligned, DynStride>
            EigenRef;

    static_assert(dim > 0, "You are a weird person");

    template< typename OtherDerived, typename OtherConstDerived,
            typename OtherSubDerived, typename OtherSuperDerived,
            typename OtherScalar, size_t OtherDim >
    friend class TensorMapBase;

protected:
    Scalar *data_;
    size_t shape_[dim];
    size_t stride_[dim];

    Derived &derived() { return *static_cast<Derived *>(this); }
    const ConstDerived &derived() const { return *static_cast<const ConstDerived *>(this); }

public:
    TensorMapBase()
     : data_(NULL)
    {}

    // Dimension constructors
    template<typename ... Dimensions>
    TensorMapBase(Scalar *data, const DynInnerStride &inner_stride, Dimensions ... dimensions)
            : data_(data)
    {
        static_assert(sizeof...(dimensions) == dim, "You passed a wrong number of dimensions");
        stride_[dim - 1] = inner_stride.inner();
        _init<0>(dimensions...);
    }

    template<typename ... Dimensions>
    TensorMapBase(Scalar *data, Dimensions ... dimensions)
            : TensorMapBase(data, DynInnerStride(1), dimensions...) {}

    // Only takes the inner stride into account ! (for now)
    template<typename ... Dimensions>
    TensorMapBase(EigenRef &&mat, Dimensions ... dimensions)
            : TensorMapBase(mat.data(), DynInnerStride(mat.innerStride()), dimensions...)
    {
        assert(shape_[0] * stride_[0] == mat.rows() * mat.cols() && "Dimensions do not match the Eigen matrix");
    }

    // Copy and move constructors
    TensorMapBase(Derived &other)
    {
        data_ = other.data_;
        std::copy(other.stride_, other.stride_ + dim, stride_);
        std::copy(other.shape_, other.shape_ + dim, shape_);
    }

    TensorMapBase(NonConst<Derived> &&other)
    {
        data_ = other.data_;
        std::copy(other.stride_, other.stride_ + dim, stride_);
        std::copy(other.shape_, other.shape_ + dim, shape_);
    }

    // Slicing construction
    template<size_t SliceDim>
    TensorMapBase(const Slice<SliceDim> &slice, SuperDerived &super)
    {
        static_assert(SliceDim <= dim, "Slice used on invalid dimension");
        assert(slice.idx < super.shape_[SliceDim] && "Index out of shape");
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy(super.stride_, super.stride_ + SliceDim, stride_);
        std::copy(super.shape_, super.shape_ + SliceDim, shape_);
        std::copy(super.stride_ + SliceDim + 1, super.stride_ + dim + 1, stride_ + SliceDim);
        std::copy(super.shape_ + SliceDim + 1, super.shape_ + dim + 1, shape_ + SliceDim);
    }

    template<size_t SliceDim>
    TensorMapBase(const Slice<SliceDim> &slice, SuperDerived &&super)
    {
        static_assert(SliceDim < dim, "Slice used on invalid dimension");
        assert(slice.idx < super.shape_[SliceDim] && "Index out of shape");
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy(super.stride_, super.stride_ + SliceDim, stride_);
        std::copy(super.shape_, super.shape_ + SliceDim, shape_);
        std::copy(super.stride_ + SliceDim + 1, super.stride_ + dim + 1, stride_ + SliceDim);
        std::copy(super.shape_ + SliceDim + 1, super.shape_ + dim + 1, shape_ + SliceDim);
    }

    // This is mainly used for binding numpy arrays
    template< typename Integer >
    TensorMapBase( Scalar* data, const Shape<Integer>& shape, const Stride<Integer>& stride )
    {
        data_ = data;
        for ( size_t i = 0 ; i < dim ; ++i )
        {
            shape_[i] = shape.shape[i];
            stride_[i] = stride.stride[i];
        }
    }

protected:
    // Used to initialize strides and shapes in constructors
    template<size_t s, typename ... OtherDimensions>
    inline void _init(size_t d, OtherDimensions ... other_dimensions)
    {
        shape_[s] = d;
        _init<s + 1>(other_dimensions...);
        stride_[s] = shape_[s + 1] * stride_[s + 1];
    }

    template<size_t s>
    inline void _init(size_t d)
    {
        shape_[s] = d;
    }

public:
    // Returns a slice, along SliceDim
    template<size_t SliceDim>
    SubDerived slice(size_t idx) const { return SubDerived(Slice<SliceDim>(idx), derived()); }

    template<size_t SliceDim>
    SubDerived slice(size_t idx) { return SubDerived(Slice<SliceDim>(idx), derived()); }


    // --- Utility methods ---

    bool ravelable() const
    {
        for (size_t i = 0; i < dim - 1; ++i)
        {
            if (shape_[i] != shape_[i + 1] * stride_[i + 1])
                return false;
        }
        return true;
    }

    size_t size() const
    {
        size_t s = 0;
        for (size_t i = 0; i < dim - 1; ++i)
            s *= shape_[i];
        return s;
    }

    Eigen::Map<Vector<NonConst<Scalar>>, Eigen::Unaligned, DynInnerStride>
    ravel()
    {
        assert(ravelable() && "Cannot be raveled");
        return Eigen::Map<Vector<NonConst<Scalar>>,
                Eigen::Unaligned, DynInnerStride>(data_, size(), DynInnerStride(stride_[dim - 1]));
    }
    Eigen::Map<const Vector<NonConst<Scalar>>, Eigen::Unaligned, DynInnerStride>
    ravel() const
    {
        assert(ravelable() && "Cannot be raveled");
        return Eigen::Map<const Vector<NonConst<Scalar>>,
                Eigen::Unaligned, DynInnerStride>(data_, size(), DynInnerStride(stride_[dim - 1]));
    }

    size_t shape(size_t i) const { return shape_[i]; }
    size_t stride(size_t i) const { return stride_[i]; }

    template< size_t NewDim, typename ... Dimensions >
    TensorMap<Scalar,NewDim>
    reshape( Dimensions ... dimensions )
    { return TensorMap<Scalar,NewDim>( data_, dimensions... ); }

    template< size_t NewDim, typename ... Dimensions >
    TensorMap<Const<Scalar>,NewDim>
    reshape( Dimensions ... dimensions ) const
    { return TensorMap<Const<Scalar>,NewDim>( data_, dimensions... ); }

    TensorMap<Scalar,dim-1>
    contractFirst()
    {
        assert(ravelable() && "Cannot be trivially contracted");
        TensorMap<Scalar,dim-1> contracted( Slice<0>(0), derived() );
        contracted.shape_[0] = shape_[0]*shape_[1];
        return contracted;
    };
    TensorMap<Const<Scalar>,dim-1>
    contractFirst() const
    {
        assert(ravelable() && "Cannot be trivially contracted");
        TensorMap<Const<Scalar>,dim-1> contracted( Slice<0>(0), derived() );
        contracted.shape_[0] = shape_[0]*shape_[1];
        return contracted;
    };

    TensorMap<Scalar,dim-1>
    contractLast()
    {
        assert(ravelable() && "Cannot be trivially contracted");
        TensorMap<Scalar,dim-1> contracted( Slice<dim-1>(0), derived() );
        contracted.shape_[dim-2] = shape_[dim-1]*shape_[dim-1];
        contracted.stride_[dim-2] = stride_[dim-1];
        return contracted;
    };
    TensorMap<Const<Scalar>,dim-1>
    contractLast() const
    {
        assert(ravelable() && "Cannot be trivially contracted");
        TensorMap<Const<Scalar>,dim-1> contracted( Slice<dim-1>(0), derived() );
        contracted.shape_[dim-2] = shape_[dim-1]*shape_[dim-1];
        contracted.stride_[dim-2] = stride_[dim-1];
        return contracted;
    };
};

// ----------------------------------------------------------------------------------------

#define TENSOR_MAP_BASE(Derived, Scalar, dim) TensorMapBase< \
    ConstAs< Scalar, Derived< NonConst<Scalar>, dim > >, \
    Const< Derived< Const<Scalar>, dim > >, \
    Derived< Scalar, dim-1 >, \
    ConstAs< Scalar, Derived< Scalar, dim+1 > >, \
    Scalar, dim >

// This class solves the const-ness issues (ex: converting 'const TensorMap<Scalar>' to 'const TensorMap<const Scalar>')
template<typename Scalar, size_t dim>
class TensorMap_ConstInterface : public TENSOR_MAP_BASE(TensorMap_ConstInterface, Scalar, dim)
{
    typedef TENSOR_MAP_BASE(TensorMap_ConstInterface, Scalar, dim) Base;

public:
    template<typename ... Args>
    explicit TensorMap_ConstInterface(Args ... args)
     : Base(args...) {}
};

} // namespace TensorMapTools

// ----------------------------------------------------------------------------------------

// This is the final class, that implements operator() and provides
// Eigen operations on dimensions 1 and 2
template< typename Scalar, size_t dim, size_t current_dim >
class TensorMap : public TensorMapTools::TensorMap_ConstInterface<Scalar,dim>
{
    static_assert( current_dim <= dim, "You probably called operator() too many times" );
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,dim> Base;

public:
    template< typename ... Args >
    TensorMap( Args ... args )
     : Base(args...)
    {}

    TensorMap< Scalar, dim, current_dim+1 >
    operator()( void )
    { return TensorMap< Scalar, dim, current_dim+1 >( *this ); };
    TensorMap< Const<Scalar>, dim, current_dim+1 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, dim, current_dim+1 >( *this ); };

    TensorMap< Scalar, dim-1, current_dim >
    operator()( size_t i )
    { return TensorMap< Scalar, dim-1, current_dim >( TensorMapTools::Slice<current_dim>(i), *this ); };
    TensorMap< Const<Scalar>, dim-1, current_dim >
    operator()( size_t i ) const
    { return TensorMap< Const<Scalar>, dim-1, current_dim >( TensorMapTools::Slice<current_dim>(i), *this ); };

    template< typename ... OtherIndices >
    typename std::result_of< TensorMap<Scalar, dim-1, current_dim >(OtherIndices...) >::type
    operator()( size_t i, OtherIndices ... indices )
    { return this->operator()(i)(indices...); }
    template< typename ... OtherIndices >
    typename std::result_of< TensorMap<Const<Scalar>, dim-1, current_dim >(OtherIndices...) >::type
    operator()( size_t i, OtherIndices ... indices ) const
    { return this->operator()(i)(indices...); }
};

template< typename Scalar >
class TensorMap<Scalar,1,0> : public TensorMapTools::TensorMap_ConstInterface<Scalar,1>,
    public Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynInnerStride >
{
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,1> Base;
    typedef Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynInnerStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
     : Base(args...),
       EigenBase( this->data_, this->shape_[0],
                  TensorMapTools::DynInnerStride(this->stride_[0]) )
    {}

    TensorMap< Scalar, 1, 1 >
    operator()( void )
    { return TensorMap< Scalar, 1, 1 >( *this ); };
    TensorMap< Const<Scalar>, 1, 1 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, 1, 1 >( *this ); };

    Scalar&
    operator()( size_t i )
    { return this->EigenBase::operator()(i); };
    Scalar
    operator()( size_t i ) const
    { return this->EigenBase::operator()(i); };
};

template< typename Scalar >
class TensorMap<Scalar,1,1> : public TensorMapTools::TensorMap_ConstInterface<Scalar,1>,
                               public Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynInnerStride >
{
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,1> Base;
    typedef Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynInnerStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
            : Base(args...),
              EigenBase( this->data_, this->shape_[0],
                         TensorMapTools::DynInnerStride(this->stride_[0]) )
    {}
};

template< typename Scalar >
class TensorMap<Scalar,2,0> : public TensorMapTools::TensorMap_ConstInterface<Scalar,2>,
                               public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride >
{
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,2> Base;
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
            : Base(args...),
              EigenBase( this->data_, this->shape_[0], this->shape_[1],
                         TensorMapTools::DynStride(this->stride_[0], this->stride_[1]) )
    {}

    TensorMap< Scalar, 2, 1 >
    operator()( void )
    { return TensorMap< Scalar, 2, 1 >( *this ); };
    TensorMap< Const<Scalar>, 2, 1 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, 2, 1 >( *this ); };

    TensorMap< Scalar, 1, 0 >
    operator()( size_t i )
    { return TensorMap< Scalar, 1, 0 >( TensorMapTools::Slice<0>(i), *this ); };
    TensorMap< Const<Scalar>, 1, 0 >
    operator()( size_t i ) const
    { return TensorMap< Const<Scalar>, 1, 0 >( TensorMapTools::Slice<0>(i), *this ); };

    Scalar&
    operator()( size_t i, size_t j )
    { return this->EigenBase::operator()( i, j ); }
    Scalar
    operator()( size_t i, size_t j ) const
    { return this->EigenBase::operator()( i, j ); }
};

template< typename Scalar >
class TensorMap<Scalar,2,1> : public TensorMapTools::TensorMap_ConstInterface<Scalar,2>,
                               public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride >
{
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,2> Base;
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
            : Base(args...),
              EigenBase( this->data_, this->shape_[0], this->shape_[1],
                         TensorMapTools::DynStride(this->stride_[0], this->stride_[1]) )
    {}

    TensorMap< Scalar, 2, 2 >
    operator()( void )
    { return TensorMap< Scalar, 2, 2 >( *this ); };
    TensorMap< Const<Scalar>, 2, 2 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, 2, 2 >( *this ); };

    TensorMap< Scalar, 1, 1 >
    operator()( size_t i )
    { return TensorMap< Scalar, 1, 1 >( TensorMapTools::Slice<1>(i), *this ); };
    TensorMap< Const<Scalar>, 1, 1 >
    operator()( size_t i ) const
    { return TensorMap< Const<Scalar>, 1, 1 >( TensorMapTools::Slice<1>(i), *this ); };
};

template< typename Scalar >
class TensorMap<Scalar,2,2> : public TensorMapTools::TensorMap_ConstInterface<Scalar,2>,
                               public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride >
{
    typedef TensorMapTools::TensorMap_ConstInterface<Scalar,2> Base;
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned,
            TensorMapTools::DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
            : Base(args...),
              EigenBase( this->data_, this->shape_[0], this->shape_[1],
                         TensorMapTools::DynStride(this->stride_[0], this->stride_[1]) )
    {}
};

#endif //TENSOR_TENSORMAP_H
