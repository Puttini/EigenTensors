#ifndef CAO_ET_AL_TENSORMAP_H
#define CAO_ET_AL_TENSORMAP_H

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

/*
// Used to convert types to const
template< typename Scalar >
struct NonConst_
{ typedef Scalar type; };
template< typename Scalar >
struct NonConst_< const Scalar >
{ typedef Scalar type; };

template< typename Scalar >
struct Const_
{ typedef const Scalar type; };
template< typename Scalar >
struct Const_< const Scalar >
{ typedef const Scalar type; };


template< typename Scalar >
using Const = typename Const_<Scalar>::type;

template< typename Scalar >
using NonConst = typename NonConst_<Scalar>::type;

template< typename AsThis, typename Type >
using ConstAs = typename std::conditional< std::is_const<AsThis>::value, Const<Type>, NonConst<Type> >::type;
*/

// ----------------------------------------------------------------------------------------

namespace TensorMapTools
{

// These little structs are used to keep easy-to-read constructors
// of TensorMapBase

template< typename Scalar, size_t dim, size_t current_dim >
class TensorMap;

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

template< size_t >
struct Contraction { };

// Dynamic strides
typedef Eigen::InnerStride<Eigen::Dynamic> DynInnerStride;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

// ----------------------------------------------------------------------------------------

#define CONDITIONAL_ENABLE_IF_TYPE( Type ) typename EnableIfType = Type
#define ENABLE_IF( cond ) typename std::enable_if< (cond) >::type* = 0

template< typename Scalar, size_t dim >
class TensorMapBase
{
    typedef Eigen::Ref< ConstAs<Scalar, MatrixR<NonConst<Scalar>>>,
            Eigen::Unaligned, DynStride >
            EigenRef;

    static_assert(dim > 0, "You are a weird person");

    template< typename OtherScalar, size_t OtherDim >
    friend class TensorMapBase;

    template< typename OtherScalar, size_t OtherDim, size_t OtherCurrentDim >
    friend class TensorMap;

protected:
    Scalar *data_;
    size_t shape_[dim];
    size_t stride_[dim];

    // --- Dimension constructors ---
private:
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
    template<typename ... Dimensions>
    TensorMapBase(Scalar *data, const DynInnerStride &inner_stride, Dimensions ... dimensions)
     : data_(data)
    {
        static_assert(sizeof...(dimensions) == dim, "You passed a wrong number of dimensions");
        stride_[dim - 1] = inner_stride.inner();
        _init<0>(dimensions...);
    }

    template<typename ... Dimensions>
    inline TensorMapBase(Scalar *data, Dimensions ... dimensions)
     : TensorMapBase(data, DynInnerStride(1), dimensions...)
    {}

    template<typename ... Dimensions>
    inline TensorMapBase( EigenRef&& mat, Dimensions ... dimensions )
     : TensorMapBase( mat.data(), DynInnerStride(mat.innerStride()), dimensions... )
    {}

    // --- Copy and move constructors ---
private:
    template< typename OtherType >
    void copy_move_constructor( OtherType other )
    {
        data_ = other.data_;
        std::copy(other.stride_, other.stride_ + dim, stride_);
        std::copy(other.shape_, other.shape_ + dim, shape_);
    }

public:
    TensorMapBase( ConstAs<Scalar,TensorMapBase<Scalar,dim>>& other )
    { copy_move_constructor< ConstAs<Scalar,TensorMapBase<Scalar,dim>>& >(other); }
    template< CONDITIONAL_ENABLE_IF_TYPE(Scalar) >
    TensorMapBase( const TensorMapBase<NonConst<Scalar>,dim>& other, ENABLE_IF(std::is_const<EnableIfType>::value) )
    { copy_move_constructor< const TensorMapBase<NonConst<Scalar>,dim>& >(other); }

    TensorMapBase( TensorMapBase<Scalar,dim>&& other )
    { copy_move_constructor< TensorMapBase<Scalar,dim>&& >( std::move(other) ); }
    template< CONDITIONAL_ENABLE_IF_TYPE(Scalar) >
    TensorMapBase( const TensorMapBase<NonConst<Scalar>,dim>&& other, ENABLE_IF(std::is_const<EnableIfType>::value) )
    { copy_move_constructor< const TensorMapBase<NonConst<Scalar>,dim>&& >( std::move(other) ); }


    // --- Slicing constructors ---
private:
    template<size_t SliceDim, typename SuperType>
    void slice_constructor( const Slice<SliceDim>& slice, SuperType super )
    {
        static_assert(SliceDim <= dim, "Slice used on invalid dimension");
        assert(slice.idx < super.shape_[SliceDim] && "Index out of shape");
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy(super.stride_, super.stride_ + SliceDim, stride_);
        std::copy(super.shape_, super.shape_ + SliceDim, shape_);
        std::copy(super.stride_ + SliceDim + 1, super.stride_ + dim + 1, stride_ + SliceDim);
        std::copy(super.shape_ + SliceDim + 1, super.shape_ + dim + 1, shape_ + SliceDim);
    }

protected:
    template<size_t SliceDim>
    TensorMapBase(const Slice<SliceDim>& slice, ConstAs<Scalar, TensorMapBase<Scalar,dim+1> >& super)
    { slice_constructor<SliceDim, ConstAs<Scalar,TensorMapBase<Scalar,dim+1>>& >(slice,super); }
    template<size_t SliceDim, CONDITIONAL_ENABLE_IF_TYPE(Scalar) >
    TensorMapBase(const Slice<SliceDim>& slice, const TensorMapBase<NonConst<Scalar>,dim+1>& super, ENABLE_IF(std::is_const<EnableIfType>::value) )
    { slice_constructor< SliceDim, const TensorMapBase<NonConst<Scalar>,dim+1>& >(slice,super); }

    template<size_t SliceDim>
    TensorMapBase(const Slice<SliceDim>& slice, TensorMapBase<Scalar,dim+1>&& super)
    { slice_constructor<SliceDim, ConstAs<Scalar,TensorMapBase<Scalar,dim+1>>&& >(slice,super); }
    template<size_t SliceDim, CONDITIONAL_ENABLE_IF_TYPE(Scalar) >
    TensorMapBase(const Slice<SliceDim>& slice, const TensorMapBase<NonConst<Scalar>,dim+1>&& super, ENABLE_IF(std::is_const<EnableIfType>::value) )
    { slice_constructor< SliceDim, const TensorMapBase<NonConst<Scalar>,dim+1>&& >(slice,super); }

    // --- Contraction construction (contracts ContractDim with ContractDim+1 dimensions) ---
private:
    template<size_t ContractDim, typename SuperType>
    void contraction_construction(SuperType super)
    {
        static_assert(ContractDim < dim, "Contraction used on invalid dimension");
        assert( super.stride_[ContractDim] == super.stride_[ContractDim+1] * super.shape_[ContractDim+1] );
        data_ = super.data_;

        std::copy(super.stride_, super.stride_ + ContractDim, stride_);
        std::copy(super.stride_ + ContractDim + 1, super.stride_ + dim + 1, stride_ + ContractDim);

        std::copy(super.shape_, super.shape_ + ContractDim, shape_);
        shape_[ContractDim] = super.shape_[ContractDim] * super.shape_[ContractDim+1];
        std::copy(super.shape_ + ContractDim + 2, super.shape_ + dim + 1, shape_ + ContractDim+1);
    }

protected:
    template<size_t ContractDim>
    TensorMapBase(const Contraction<ContractDim>&, ConstAs<Scalar,TensorMapBase<NonConst<Scalar>,dim+1>>& super)
    { contraction_construction< ContractDim, ConstAs<Scalar,TensorMapBase<NonConst<Scalar>,dim+1>>& >(super); }
    template<size_t ContractDim, CONDITIONAL_ENABLE_IF_TYPE(Scalar)>
    TensorMapBase(const Contraction<ContractDim>&, const TensorMapBase<Scalar,dim+1>& super, ENABLE_IF(std::is_const<EnableIfType>::value) )
    { contraction_construction< ContractDim, const TensorMapBase<Scalar,dim+1>& >(super); }

public:
    // This is a bit dirty, but mainly used for binding numpy arrays
    template< typename Integer >
    TensorMapBase( Scalar* data, const Shape<Integer>& shape, const Stride<Integer>& stride )
    {
        data_ = data;
        for ( size_t i = 0 ; i < dim ; ++i )
        {
            shape_[i] = shape.shape[i];
            stride_[i] = stride.stride[i] / sizeof(Scalar);
        }
    }

public:
    // Returns a slice, along SliceDim
    template<size_t SliceDim>
    TensorMap<Const<Scalar>,dim-1,0> slice(size_t idx) const
    { return TensorMap<Const<Scalar>,dim-1,0>(Slice<SliceDim>(idx), *this); }
    template<size_t SliceDim>
    TensorMap<Scalar,dim-1,0> slice(size_t idx)
    { return TensorMap<Scalar,dim-1,0>(Slice<SliceDim>(idx), *this); }


    // --- Utility methods ---

    bool ravelable() const
    {
        for (size_t i = 0; i < dim - 1; ++i)
        {
            if (stride_[i] != shape_[i + 1] * stride_[i + 1])
                return false;
        }
        return true;
    }

    size_t size() const
    {
        size_t s = 1;
        for (size_t i = 0; i < dim; ++i)
            s *= shape_[i];
        return s;
    }

    Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, DynInnerStride >
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
    TensorMap<Scalar,NewDim,0>
    reshape( Dimensions ... dimensions )
    { return TensorMap<Scalar,NewDim,0>( data_, dimensions... ); }

    template< size_t NewDim, typename ... Dimensions >
    TensorMap<Const<Scalar>,NewDim,0>
    reshape( Dimensions ... dimensions ) const
    { return TensorMap<Const<Scalar>,NewDim,0>( data_, dimensions... ); }

    // Contracts ContractDim with ContractDim+1 dimensions
    template< size_t ContractDim >
    TensorMap<Scalar,dim-1,0>
    contract()
    {
        assert( stride_[ContractDim] == stride_[ContractDim+1] * shape_[ContractDim+1]
                && "Cannot be trivially contracted");
        return TensorMap<Scalar,dim-1,0>( Contraction<ContractDim>(), *this );
    };
    template< size_t ContractDim >
    TensorMap<Const<Scalar>,dim-1,0>
    contract() const
    {
        assert( stride_[ContractDim] == stride_[ContractDim+1] * shape_[ContractDim+1]
                && "Cannot be trivially contracted");
        return TensorMap<Const<Scalar>,dim-1,0>( Contraction<ContractDim>(), *this );
    };

    TensorMap<Scalar,dim-1,0> contractFirst()
    { return contract<0>(); };
    TensorMap<Const<Scalar>,dim-1,0> contractFirst() const
    { return contract<0>(); };

    TensorMap<Scalar,dim-1,0> contractLast()
    { return contract<dim-2>(); };
    TensorMap<Const<Scalar>,dim-1,0> contractLast() const
    { return contract<dim-2>(); };
};

// ----------------------------------------------------------------------------------------

// This is the final class, that implements operator() and provides
// Eigen operations on dimensions 1 and 2
template< typename Scalar, size_t dim, size_t current_dim >
class TensorMap : public TensorMapBase<Scalar,dim>
{
    static_assert( current_dim <= dim, "You probably called operator() too many times" );

public:
    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMapBase<Scalar,dim>(args...)
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
class TensorMap<Scalar,1,0> :
        public TensorMapBase<Scalar,1>,
        public Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, DynInnerStride >
{
    typedef Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, DynInnerStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMapBase<Scalar,1>(args...),
       EigenBase( this->data_, this->shape_[0],
                  DynInnerStride(this->stride_[0]) )
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
class TensorMap<Scalar,1,1> :
        public TensorMapBase<Scalar,1>,
        public Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, DynInnerStride >
{
    typedef Eigen::Map< ConstAs<Scalar,Vector<NonConst<Scalar>>>, Eigen::Unaligned, DynInnerStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
    : TensorMapBase<Scalar,1>(args...),
      EigenBase( this->data_, this->shape_[0],
                 TensorMapTools::DynInnerStride(this->stride_[0]) )
    {}
};

template< typename Scalar >
class TensorMap<Scalar,2,0> :
        public TensorMapBase<Scalar,2>,
        public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, DynStride >
{
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMapBase<Scalar,2>(args...),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    TensorMap< Scalar, 2, 1 >
    operator()( void )
    { return TensorMap< Scalar, 2, 1 >( *this ); };
    TensorMap< Const<Scalar>, 2, 1 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, 2, 1 >( *this ); };

    TensorMap< Scalar, 1, 0 >
    operator()( size_t i )
    { return TensorMap< Scalar, 1, 0 >( Slice<0>(i), *this ); };
    TensorMap< Const<Scalar>, 1, 0 >
    operator()( size_t i ) const
    { return TensorMap< Const<Scalar>, 1, 0 >( Slice<0>(i), *this ); };

    Scalar&
    operator()( size_t i, size_t j )
    { return this->EigenBase::operator()( i, j ); }
    Scalar
    operator()( size_t i, size_t j ) const
    { return this->EigenBase::operator()( i, j ); }
};

template< typename Scalar >
class TensorMap<Scalar,2,1> :
        public TensorMapBase<Scalar,2>,
        public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride >
{
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, TensorMapTools::DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMapBase<Scalar,2>(args...),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    TensorMap< Scalar, 2, 2 >
    operator()( void )
    { return TensorMap< Scalar, 2, 2 >( *this ); };
    TensorMap< Const<Scalar>, 2, 2 >
    operator()( void ) const
    { return TensorMap< Const<Scalar>, 2, 2 >( *this ); };

    TensorMap< Scalar, 1, 1 >
    operator()( size_t i )
    { return TensorMap< Scalar, 1, 1 >( Slice<1>(i), *this ); };
    TensorMap< Const<Scalar>, 1, 1 >
    operator()( size_t i ) const
    { return TensorMap< Const<Scalar>, 1, 1 >( Slice<1>(i), *this ); };
};

template< typename Scalar >
class TensorMap<Scalar,2,2> :
        public TensorMapBase<Scalar,2>,
        public Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, DynStride >
{
    typedef Eigen::Map< ConstAs<Scalar,MatrixR<NonConst<Scalar>>>, Eigen::Unaligned, DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;

    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMapBase<Scalar,2>(args...),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  TensorMapTools::DynStride(this->stride_[0], this->stride_[1]) )
    {}
};

} // namespace TensorMapTools

template< typename Scalar, size_t dim >
using TensorMap = TensorMapTools::TensorMap<Scalar,dim,0>;

#endif //CAO_ET_AL_TENSORMAP_H
