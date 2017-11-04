#include <iostream>
#include <Eigen/Eigen>

template< typename Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixR = Eigen::Matrix< Scalar, rows, cols, Eigen::RowMajor >;

template< typename Scalar, int rows = Eigen::Dynamic >
using Vector = Eigen::Matrix< Scalar, rows, 1 >;

template< typename Scalar >
using _MatrixR = Eigen::Ref< MatrixR<Scalar> >;

// -----------------------------------------------------------------------------------

template< typename Scalar, int dim >
struct TensorMapExp
{
    Scalar* data;
    int* stride;
    int* shape;

// -----------------------------------------------------------------------------------

    TensorMapExp( Scalar* data, int* stride, int* shape )
     : data(data), stride(stride), shape(shape)
    {}
};

template< typename Scalar, int dim >
struct SliceTensorMapExp : public TensorMapExp<Scalar,dim-1>
{
    using TensorMapExp<Scalar,dim-1>::operator=;
    using TensorMapExp<Scalar,dim-1>::operator<<;

    template< typename Derived >
    SliceTensorMapExp( Derived& derived, int i )
     : TensorMapExp<Scalar,dim-1>( derived.data + i*derived.stride[0], derived.stride+1, derived.shape+1 )
    {
        assert( i < derived.shape[0] );
    }
};

template< typename Scalar, int dim >
struct TensorMap
{
    Scalar* data;
    int stride[dim];
    int shape[dim];

// -----------------------------------------------------------------------------------

    template< typename ... Dimensions >
    TensorMap( _MatrixR<Scalar> mat, Dimensions ... dimensions )
    {
        data = mat.data();
        static_assert( sizeof...(dimensions) == dim, "Invalid dimensions" );
        stride[dim-1] = 1;
        _init<0>( dimensions... );
    }

    template< int s, typename ... Dimensions >
    void _init( int d, Dimensions ... dimensions )
    {
        shape[s] = d;
        _init<s+1>( dimensions... );
        stride[s] = stride[s+1]*shape[s+1];
    }

    template< int s >
    void _init( int d )
    { shape[s] = d; }

    SliceTensorMapExp<Scalar,dim> operator()( int i )
    { return SliceTensorMapExp<Scalar,dim>( *this, i ); }

    template< typename ... Dimensions >
    typename std::result_of< SliceTensorMapExp<Scalar,dim>(Dimensions...) >::type
    operator()( int i, Dimensions ... dimensions )
    { return this->operator()(i)(dimensions...); };
};

template< typename Scalar >
struct TensorMapExp<Scalar,2> : public Eigen::Map< MatrixR<Scalar> >
{
    typedef Eigen::Map< MatrixR<Scalar> > EigenEquivalent;

    using EigenEquivalent::operator=;
    using EigenEquivalent::operator<<;

    Scalar* data;
    int* stride;
    int* shape;

// -----------------------------------------------------------------------------------

    TensorMapExp( Scalar* data, int* stride, int* shape )
     : EigenEquivalent( data, shape[0], shape[1] ),
       data(data), stride(stride), shape(shape)
    {
        assert( stride[0] == shape[1] && stride[1] == 1 );
    }

    SliceTensorMapExp<Scalar,2> operator()( int i )
    { return SliceTensorMapExp<Scalar,2>( *this, i ); }

    typename std::result_of< EigenEquivalent(int,int) >::type
    operator()( int i, int j )
    { return this->EigenEquivalent::operator()(i,j); }
};

template< typename Scalar >
struct TensorMapExp<Scalar,1> : public Eigen::Map< Vector<Scalar> >
{
    typedef Eigen::Map< Vector<Scalar> > EigenEquivalent;

    using EigenEquivalent::operator=;
    using EigenEquivalent::operator<<;

// -----------------------------------------------------------------------------------

    TensorMapExp( Scalar* data, int* stride, int* shape )
     : EigenEquivalent( data, shape[0] )
    {
        assert( stride[0] == 1 );
    }
};

int main()
{
    MatrixR<float,3*3,4> m;
    m.setZero();
    TensorMap<float,3> t( m, 3, 3, 4 );

    t(0,0,0) = 1;
    t(0,0)(1) = 2;
    t(0)(0,2) = 3;
    t(0)(0)(3) = 4;

    t(0,1) = Vector<float,2>( 5, 6 );
    t(0)(2) = Vector<float,2>( 5, 6 );

    return 0;
}