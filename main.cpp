#include <iostream>
#include "ConstScheme1.h"
#include "ConstScheme2.h"
#include "InheritScheme1.h"

#include "TensorMap.h"

template< typename Scalar >
using ConstScheme = ConstScheme3<Scalar>;

template< typename Scalar >
using InheritScheme = InheritScheme1<Scalar>;

void f1( const int& i )
{}
void f2( int i )
{}
void f3( int& i )
{}

void test_schemes()
{
    // Const tests
    int i = 0;
    ConstScheme<int> a(&i);
    ConstScheme<const int> b(&i);
    ConstScheme<int> c(a);
    ConstScheme<const int> d(b);
    ConstScheme<const int> e(a);
    ConstScheme<const int> g( a.copy() );
    ConstScheme<int> h( a.copy().copy() );
    ConstScheme<int> j = ConstScheme<int>(a);
    j.f() = 3;

    f1( a.f() );
    f1( b.f() );
    f2( a.f() );
    f2( b.f() );
    f3( a.f() );

    // Inheritance tests
    InheritScheme<int> x1(&i);
    InheritScheme<int> x2(x1);
    InheritScheme<const int> x3(x1);
    InheritScheme<int> x4( a );
}

int main( int argc, char** argv )
{
    test_schemes();

    // Const and construction tests
    MatrixR<float,4,4> m;
    TensorMap<float,3> t( m, 4, 2, 2 );
    Eigen::Map< const MatrixR<float,4,4> > const_map( m.data() );
    TensorMap<float,3> t1( m.data(), 4, 2, 2 );
    const TensorMap<const float,3> t2( m.data(), 4, 2, 2 );
    TensorMap<const float,3> t3( const_map.data(), InnerStride(1), 4, 2, 2 );

    //TensorMap<float,3> t4( const_map.data(), 4, 2, 2 ); // Error
    //TensorMap<float,3> t5( m, 4, 2, 3 ); // Runtime Error
    //TensorMap<float,4> t6( m, 4, 2, 2 ); // Error
    //TensorMap<float,0> t7( m ); // Error
    //TensorMap<float,2> sub1( t ); // Error

    TensorMap<float,3> t8(t);
    TensorMap<const float,3> t9 = TensorMap<float,3>(t);

    // Slicing
    //TensorMap<float,2> sub2( Slice<2>(0), t );  // Error
    //t.slice<1>( 3 ); // Runtime error
    TensorMap<float,2> sub3 = t.slice<1>(1);
    TensorMap<const float,2> sub4 = t.slice<1>(1);
    t2.slice<1>(1);
    TensorMap<const float,2> sub5 = t2.slice<0>(0);

    // operator()
    auto o1 = t();
    auto o2 = t()();
    auto o3 = t()()();
    TensorMap<float,3> r1 = o3;
    TensorMap<const float,3> r2( o3 );
    const TensorMap<const float,3> r3( o2 );
    o2();
    r3()()();
    //auto o4 = t()()()(); // Error

    return 0;
}
