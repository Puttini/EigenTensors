#ifndef TENSOR_TENSORMAP_H
#define TENSOR_TENSORMAP_H

#define USE_EXTENDED_CONSTRUCTOR( Derived, Base ) \
    template< typename ... Args > \
    Derived( Args ... args ) \
     : Base( args... ) \
    {}

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

// ----------------------------------------------------------------------------------------

// Used to initialize strides and shapes from variadic parameters
template< size_t dim >
struct InitStrideAndShape
{
    template< typename ... OtherDimensionsAndStride >
    static void init(size_t *stride, size_t *shape, size_t d, OtherDimensionsAndStride... args)
    {
        shape[0] = d;
        InitStrideAndShape<dim-1>::init(stride+1, shape+1, args... );
        stride[0] = stride[1] * shape[1];
    }
};
template<>
struct InitStrideAndShape<1>
{
    static void init(size_t *stride, size_t *shape, size_t d, size_t innerStride = 1)
    {
        shape[0] = d;
        stride[0] = innerStride;
    }
};

// ----------------------------------------------------------------------------------------

// These structures are used to help using big matrices of high dimensionality
// There is much to improve, but it is still useful for now
//
// It is more user friendly than Eigen's Tensor structes (when it works)
// Moreover, I had some troubles building Eigen's Tensor structes on Windows...

template< typename Scalar, size_t dim, size_t current_dim = 0 >
struct TensorMap_Dim;

struct Slice {};

template< typename Scalar_, size_t dim >
struct TensorMapBase
{
    typedef NonConst<Scalar_> Scalar;

    typedef Eigen::Ref< ConstAs<Scalar_,MatrixR<Scalar>>, Eigen::Unaligned,
            Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> >
            EigenRef;
    typedef Eigen::InnerStride<Eigen::Dynamic> Stride;

    static_assert( dim > 0, "That's weird" );

    Scalar_* data_;
    size_t shape_[dim];
    size_t stride_[dim];

    // This constructor skips the 'current_dim-1' coordinate in strides and shapes
    // This is used for slicing and other kinds of operations
    TensorMapBase( Slice, size_t i, size_t current_dim, ConstAs<Scalar_,TensorMapBase<Scalar_, dim>>& other )
    {
        data_ = other.data_ + i*other.stride_[current_dim];
        if ( current_dim > 0 )
        {
            std::copy( other.shape_, other.shape_ + current_dim, shape_);
            std::copy( other.stride_, other.stride_ + current_dim, stride_);
        }

        std::copy( other.shape_+current_dim, other.shape_+dim, shape_+current_dim );
        std::copy( other.stride_+current_dim, other.stride_+dim, stride_+current_dim );
    }
    TensorMapBase( Slice, size_t i, size_t current_dim, ConstAs<Scalar_,TensorMapBase<Scalar_, dim+1>>& other )
    {
        data_ = other.data_ + i*other.stride_[current_dim];
        if ( current_dim > 0 )
        {
            std::copy( other.shape_, other.shape_ + current_dim, shape_);
            std::copy( other.stride_, other.stride_ + current_dim, stride_);
        }

        std::copy( other.shape_+current_dim+1, other.shape_+dim+1, shape_+current_dim );
        std::copy( other.stride_+current_dim+1, other.stride_+dim+1, stride_+current_dim );
    }
    TensorMapBase( Slice, size_t i, size_t current_dim, ConstAs<Scalar_,TensorMapBase<Scalar_, dim>>&& other )
    {
        data_ = other.data_ + i*other.stride_[current_dim];
        if ( current_dim > 0 )
        {
            std::copy( other.shape_, other.shape_ + current_dim, shape_);
            std::copy( other.stride_, other.stride_ + current_dim, stride_);
        }

        std::copy( other.shape_+current_dim, other.shape_+dim+1, shape_+current_dim );
        std::copy( other.stride_+current_dim, other.stride_+dim+1, stride_+current_dim );
    }

    TensorMapBase( Slice, size_t current_dim, ConstAs<Scalar_,TensorMapBase<Scalar_, dim>>& other )
     : TensorMapBase( Slice(), 0, current_dim, other )
    {}

    template< typename ... DimensionsAndInnerStride >
    TensorMapBase( Scalar_* data, DimensionsAndInnerStride... args )
    {
        data_ = data;
        InitStrideAndShape<dim>::init( stride_, shape_, args... );
    }

    // Note that it will not take the outer stride into account ! (for now)
    template< typename ... Dimensions >
    TensorMapBase( EigenRef mat, Dimensions... dimensions )
     : TensorMapBase( mat.data(), dimensions..., mat.innerStride() )
    {}

    TensorMapBase( ConstAs<Scalar_,TensorMapBase<Scalar_, dim>>& other )
     : TensorMapBase( Slice(), 0, 0, other )
    {}
    TensorMapBase( ConstAs<Scalar_,TensorMapBase<Scalar_, dim+1>>& other )
     : TensorMapBase( Slice(), 0, 0, other )
    {}

    TensorMapBase( ConstAs<Scalar_,TensorMapBase<Scalar_, dim>>&& other )
     : TensorMapBase( Slice(), 0, 0, other )
    {}
    TensorMapBase( ConstAs<Scalar_,TensorMapBase<Scalar_, dim+1>>&& other )
     : TensorMapBase( Slice(), 0, 0, other )
    {}

    size_t shape( size_t i ) const
    { return shape_[i]; }
    size_t stride( size_t i ) const
    { return stride_[i]; }

    Eigen::Map< Vector<Scalar>, Eigen::Unaligned, Stride >
    ravel()
    {
        ulong new_shape = 1;
        for ( ulong i = 0 ; i < dim ; ++i )
        {
            assert( i == dim-1 || stride_[i] == shape_[i+1] * stride_[i+1] ); // Impossible to ravel !
            new_shape *= shape_[i];
        }

        return Eigen::Map< Vector<Scalar>, Eigen::Unaligned, Stride >(
                data_, new_shape, Stride(stride_[dim-1]) );
    }
    Eigen::Map< const Vector<Scalar>, Eigen::Unaligned, Stride >
    ravel() const
    {
        ulong new_shape = 1;
        for ( ulong i = 0 ; i < dim ; ++i )
        {
            assert( i == dim-1 || stride_[i] == shape_[i+1] * stride_[i+1] ); // Impossible to ravel !
            new_shape *= shape_[i];
        }

        return Eigen::Map< const Vector<Scalar>, Eigen::Unaligned, Stride >(
                data_, new_shape, Stride(stride_[dim-1]) );
    }

    template< typename ... Dimensions >
    TensorMap_Dim< Scalar_, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions )
    {
        for ( ulong i = 0 ; i < dim-1 ; ++i )
            assert(stride_[i] == shape_[i + 1] * stride_[i + 1]); // Impossible to ravel !
        return TensorMap_Dim< Scalar_, sizeof...(Dimensions) >( data_, dimensions..., stride_[dim-1] );
    }
    template< typename ... Dimensions >
    TensorMap_Dim< Scalar_, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions ) const
    {
        for ( ulong i = 0 ; i < dim-1 ; ++i )
            assert(stride_[i] == shape_[i + 1] * stride_[i + 1]); // Impossible to ravel !
        return TensorMap_Dim< Scalar_, sizeof...(Dimensions) >( data_, dimensions..., stride_[dim-1] );
    }

    TensorMap_Dim< Scalar_, dim-1 >
    contractFirst()
    {
        assert( stride_[0] == stride_[1] * shape_[1] );
        TensorMap_Dim<Scalar_,dim-1> t( 0, *this );
        t.shape_[0] = shape_[0];

        return t;
    }
    TensorMap_Dim< Const<Scalar>, dim-1 >
    contractFirst() const
    {
        assert( stride_[0] == stride_[1] * shape_[1] );
        TensorMap_Dim<Const<Scalar>,dim-1> t( TensorMap_Dim<Const<Scalar>,dim>(*this) );
        t.shape_[0] = shape_[0];

        return t;
    }

    TensorMap_Dim< Scalar, dim-1 >
    contractLast()
    {
        assert( stride_[dim-2] == stride_[dim-1] * shape_[dim-1] );
        TensorMap_Dim<Scalar,dim-1> t( dim-1, *this );
        t.stride_[dim-2] = stride_[dim-1];
        t.shape_[dim-2] = shape_[dim-1];
        return t;
    }
    TensorMap_Dim< Const<Scalar>, dim-1 >
    contractLast() const
    {
        assert( stride_[dim-2] == stride_[dim-1] * shape_[dim-1] );
        TensorMap_Dim<Const<Scalar>,dim-1> t( dim-1, *this );
        t.stride_[dim-2] = stride_[dim-1];
        t.shape_[dim-2] = shape_[dim-1];
        return t;
    }

    template< size_t dimension >
    TensorMap_Dim<Scalar,dim-1>
    access( size_t i )
    { return TensorMap_Dim<Scalar,dim-1>( Slice(), i, dimension, *this ); }
    template< size_t dimension >
    TensorMap_Dim<Const<Scalar>,dim-1>
    access( size_t i ) const
    { return TensorMap_Dim<Const<Scalar>,dim-1>( Slice(), i, dimension, *this ); }
};

// ----------------------------------------------------------------------------------------

// This struct is used to add convertion from 'const TensorMapBase<Scalar,dim>&'
// to 'const TensorMapBase<const Scalar,dim>&'
template< typename Scalar, size_t dim >
struct TensorMapBase_ConstInterface : public TensorMapBase<Scalar,dim>
{
    typedef TensorMapBase<Scalar,dim> Base;
    USE_EXTENDED_CONSTRUCTOR( TensorMapBase_ConstInterface, Base )
};

template< typename Scalar, size_t dim >
struct TensorMapBase_ConstInterface< const Scalar, dim > : public TensorMapBase< const Scalar, dim >
{
    typedef TensorMapBase<const Scalar,dim> Base;
    typedef TensorMapBase<Scalar,dim> NonConstBase;
    TensorMapBase_ConstInterface( size_t i, size_t current_dim, const NonConstBase& other )
     : TensorMapBase<const Scalar,dim>( i, current_dim, *reinterpret_cast<const Base*>(&other) )
    {}
    TensorMapBase_ConstInterface( size_t current_dim, const NonConstBase& other )
     : TensorMapBase<const Scalar,dim>( 0, current_dim, *reinterpret_cast<const Base*>(&other) )
    {}
    TensorMapBase_ConstInterface( const NonConstBase& other )
     : TensorMapBase<const Scalar,dim>( 0, 0, *reinterpret_cast<const Base*>(&other) )
    {}

    USE_EXTENDED_CONSTRUCTOR( TensorMapBase_ConstInterface, Base )
};

// ----------------------------------------------------------------------------------------

// This defines (if possible) the Eigen interface with Matrices and Vectors
template< typename Scalar_, size_t dim >
struct TensorMapBase_EigenInterface : public TensorMapBase_ConstInterface<Scalar_,dim>
{
    typedef TensorMapBase_ConstInterface<Scalar_,dim> Base;
    USE_EXTENDED_CONSTRUCTOR( TensorMapBase_EigenInterface, Base )
};

template< typename Scalar_ >
struct TensorMapBase_EigenInterface<Scalar_,2>
        : public TensorMapBase_ConstInterface<Scalar_,2>,
          public Eigen::Map< ConstAs<Scalar_,MatrixR<NonConst<Scalar_>>>, Eigen::Unaligned,
                  Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> >
{
    typedef NonConst<Scalar_> Scalar;
    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;
    typedef Eigen::Map< ConstAs<Scalar_,MatrixR<Scalar>>,
            Eigen::Unaligned, Stride > EigenEquivalent;

    using EigenEquivalent::operator();
    using EigenEquivalent::operator=;
    using EigenEquivalent::operator-=;
    using EigenEquivalent::operator+=;

    template< typename ... Args >
    TensorMapBase_EigenInterface<Scalar_,2>( Args ... args )
     : TensorMapBase_ConstInterface<Scalar_,2>( args... ),
       EigenEquivalent( this->data_, this->shape_[0], this->shape_[1],
                        Stride(this->stride_[0], this->stride_[1]) )
    {}
};

template< typename Scalar_ >
struct TensorMapBase_EigenInterface<Scalar_,1>
        : public TensorMapBase_ConstInterface<Scalar_,1>,
          public Eigen::Map< ConstAs<Scalar_,Vector<NonConst<Scalar_>>>, Eigen::Unaligned,
                  Eigen::InnerStride<Eigen::Dynamic> >
{
    typedef NonConst<Scalar_> Scalar;
    typedef Eigen::InnerStride<Eigen::Dynamic> Stride;
    typedef Eigen::Map< ConstAs<Scalar_,Vector<NonConst<Scalar>>>,
            Eigen::Unaligned, Stride > EigenEquivalent;

    using EigenEquivalent::operator();
    using EigenEquivalent::operator=;
    using EigenEquivalent::operator-=;
    using EigenEquivalent::operator+=;

    template< typename ... Args >
    TensorMapBase_EigenInterface<Scalar_,1>( Args ... args )
     : TensorMapBase_ConstInterface<Scalar_,1>( args... ),
       EigenEquivalent( this->data_, this->shape_[0],
                        Stride(this->stride_[0]) )
    {}
};

// ----------------------------------------------------------------------------------------

// This one is used to specify (or not) an operator() given the 'current_dim'
template< typename Scalar_, size_t dim, size_t current_dim >
struct TensorMap_Dim : public TensorMapBase_EigenInterface<Scalar_,dim>
{
    static_assert( current_dim <= dim, "You probably called operator() too many times" );
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
     : TensorMapBase_EigenInterface<Scalar_,dim>(args...)
    {}

    TensorMap_Dim<Const<Scalar>,dim,current_dim+1>
    operator()( void ) const
    { return TensorMap_Dim<Const<Scalar>,dim,current_dim+1>(*this); }

    TensorMap_Dim<Scalar,dim,current_dim+1>
    operator()( void )
    { return TensorMap_Dim<Scalar,dim,current_dim+1>(*this); }

    TensorMap_Dim<Const<Scalar>,dim-1,current_dim>
    operator()( size_t i ) const
    { return TensorMap_Dim<Const<Scalar>,dim-1,current_dim>( Slice(), i, current_dim, *this); };

    TensorMap_Dim<Scalar,dim-1,current_dim>
    operator()( size_t i )
    { return TensorMap_Dim<Scalar,dim-1,current_dim>( Slice(), i, current_dim, *this ); };

    template< typename ... OtherIndices >
    typename std::result_of< TensorMap_Dim<Scalar,dim-1,current_dim>( OtherIndices... ) >::type
    operator()( size_t i, OtherIndices... otherIndices )
    { return TensorMap_Dim<Scalar,dim-1,current_dim>( Slice(), i, current_dim, *this)( otherIndices... ); };

    template< typename ... OtherIndices >
    typename std::result_of< TensorMap_Dim<Const<Scalar>,dim-1,current_dim>( OtherIndices... ) >::type
    operator()( size_t i, OtherIndices... otherIndices ) const
    { return TensorMap_Dim<Const<Scalar>,dim-1,current_dim>( Slice(), i, current_dim, *this)( otherIndices... ); };
};

// Matrix specializations
template< typename Scalar_ >
struct TensorMap_Dim<Scalar_,2,0> : public TensorMapBase_EigenInterface<Scalar_,2>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
     : TensorMapBase_EigenInterface<Scalar_,2>( args... )
    {}

    using TensorMapBase_EigenInterface<Scalar_,2>::operator=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator+=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator-=;

    //TensorMap_Dim<Const<Scalar>,2,1>
    //operator()( void ) const
    //{ return TensorMap_Dim<Const<Scalar>,2,1>(*this); }

    TensorMap_Dim<Scalar,2,1>
    operator()( void )
    { return TensorMap_Dim<Scalar,2,1>(*this); }

    //TensorMap_Dim<Const<Scalar>,1,0>
    //operator()( size_t i ) const
    //{ return TensorMap_Dim<Const<Scalar>,1,0>( i, 0, *this); };

    TensorMap_Dim<Scalar,1,0>
    operator()( size_t i )
    { return TensorMap_Dim<Scalar,1,0>( Slice(), i, 0, *this); };

    Scalar_&
    operator()( size_t i, size_t j )
    { return this->TensorMapBase_EigenInterface<Scalar_,2>::operator()(i,j); }
};

template< typename Scalar_ >
struct TensorMap_Dim<Scalar_,2,1> : public TensorMapBase_EigenInterface<Scalar_,2>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
     : TensorMapBase_EigenInterface<Scalar_,2>( args... )
    {}

    using TensorMapBase_EigenInterface<Scalar_,2>::operator=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator+=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator-=;

    //TensorMap_Dim<Const<Scalar>,2,2>
    //operator()( void ) const
    //{ return TensorMap_Dim<Const<Scalar>,2,2>(*this); }

    TensorMap_Dim<Scalar,2,2>
    operator()( void )
    { return TensorMap_Dim<Scalar,2,2>(*this); }

    //TensorMap_Dim<Const<Scalar>,1,1>
    //operator()( size_t i ) const
    //{ return TensorMap_Dim<Const<Scalar>,1,1>( i, 1, *this); };

    TensorMap_Dim<Scalar,1,1>
    operator()( size_t i )
    { return TensorMap_Dim<Scalar,1,1>( Slice(), i, 1, *this); };
};

template< typename Scalar_ >
struct TensorMap_Dim<Scalar_,2,2> : public TensorMapBase_EigenInterface<Scalar_,2>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
            : TensorMapBase_EigenInterface<Scalar_,2>( args... )
    {}

    using TensorMapBase_EigenInterface<Scalar_,2>::operator=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator+=;
    using TensorMapBase_EigenInterface<Scalar_,2>::operator-=;
};


// Vector specializations
template< typename Scalar_ >
struct TensorMap_Dim<Scalar_,1,0> : public TensorMapBase_EigenInterface<Scalar_,1>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
            : TensorMapBase_EigenInterface<Scalar_,1>( args... )
    {}

    using TensorMapBase_EigenInterface<Scalar_,1>::operator();
    using TensorMapBase_EigenInterface<Scalar_,1>::operator=;
    using TensorMapBase_EigenInterface<Scalar_,1>::operator+=;
    using TensorMapBase_EigenInterface<Scalar_,1>::operator-=;

    TensorMap_Dim<Const<Scalar>,1,1>
    operator()( void ) const
    { return TensorMap_Dim<Const<Scalar>,1,1>(*this); }

    TensorMap_Dim<Scalar,1,1>
    operator()( void )
    { return TensorMap_Dim<Scalar,1,1>(*this); }
};

template< typename Scalar_ >
struct TensorMap_Dim<Scalar_,1,1> : public TensorMapBase_EigenInterface<Scalar_,1>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap_Dim( Args ... args )
            : TensorMapBase_EigenInterface<Scalar_,1>( args... )
    {}

    using TensorMapBase_EigenInterface<Scalar_,1>::TensorMapBase_EigenInterface;
    using TensorMapBase_EigenInterface<Scalar_,1>::operator=;
    using TensorMapBase_EigenInterface<Scalar_,1>::operator+=;
    using TensorMapBase_EigenInterface<Scalar_,1>::operator-=;
};

// ----------------------------------------------------------------------------------------

// Finally, the user-friendly pretty struct
template< typename Scalar_, size_t dim >
struct TensorMap : public TensorMap_Dim<Scalar_,dim,0>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap( Args ... args )
     : TensorMap_Dim<Scalar_,dim,0>( args... )
    {}

    using TensorMap_Dim<Scalar_,dim,0>::TensorMap_Dim;
    using TensorMap_Dim<Scalar_,dim,0>::operator();

    TensorMap()
            : TensorMap_Dim<Scalar_,dim,0>( NULL, 0, 0 )
    {}
};

template< typename Scalar_ >
struct TensorMap<Scalar_,2> : public TensorMap_Dim<Scalar_,2,0>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap( Args ... args )
            : TensorMap_Dim<Scalar_,2,0>( args... )
    {}

    using TensorMap_Dim<Scalar_,2,0>::operator();
    using TensorMap_Dim<Scalar_,2,0>::operator=;
    using TensorMap_Dim<Scalar_,2,0>::operator+=;
    using TensorMap_Dim<Scalar_,2,0>::operator-=;

    TensorMap()
            : TensorMap_Dim<Scalar_,2,0>( NULL, 0, 0 )
    {}
};

template< typename Scalar_ >
struct TensorMap<Scalar_,1> : public TensorMap_Dim<Scalar_,1,0>
{
    typedef NonConst<Scalar_> Scalar;

    template< typename ... Args >
    TensorMap( Args ... args )
            : TensorMap_Dim<Scalar_,1,0>( args... )
    {}

    using TensorMap_Dim<Scalar_,1,0>::operator();
    using TensorMap_Dim<Scalar_,1,0>::operator=;
    using TensorMap_Dim<Scalar_,1,0>::operator+=;
    using TensorMap_Dim<Scalar_,1,0>::operator-=;

    TensorMap()
            : TensorMap_Dim<Scalar_,1,0>( NULL, 0, 0 )
    {}
};

#endif //TENSOR_TENSORMAP_H
