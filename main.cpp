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

int main_old( int argc, char** argv )
{
    test_schemes();

    // Const and construction tests
    MatrixR<float,4,4> m = MatrixR<float,4,4>::Constant(42);
    Eigen::Map< const MatrixR<float,4,4> > const_map( m.data() );
    TensorMap<float,3> t1( m.data(), 4, 2, 2 );
    TensorMap<float,3> t( m.data(), Eigen::InnerStride<>(m.innerStride()), 4, 2, 2 );
    const TensorMap<const float,3> t2( m.data(), 4, 2, 2 );
    TensorMap<const float,3> t3( const_map.data(), Eigen::InnerStride<>(1), 4, 2, 2 );

    //TensorMap<float,3> t4( const_map.data(), 4, 2, 2 ); // Error
    //TensorMap<float,3> t5( m, 4, 2, 3 ); // Runtime Error
    //TensorMap<float,4> t6( m, 4, 2, 2 ); // Error
    //TensorMap<float,0> t7( m ); // Error
    //TensorMap<float,2> sub1( t ); // Error

    TensorMap<float,3> t8(t);
    TensorMap<const float,3> t9 = TensorMap<float,3>(t);

    // Slicing
    //TensorMap<float,2> sub2( Slice<3>(0), t );  // Error
    //t.slice<1>( 3 ); // Runtime error
    TensorMap<float,2> sub3 = t.slice<1>(1);
    TensorMap<const float,2> sub4 = t.slice<1>(1);
    t2.slice<1>(1);
    TensorMap<const float,2> sub5 = t2.slice<0>(0);

    // operator()( void )
    auto o1 = t();
    auto o2 = t()();
    auto o3 = t()()();
    TensorMap<float,3> r1 = o3;
    TensorMap<const float,3> r2( o3 );
    const TensorMap<const float,3> r3( o2 );
    o2();
    r3()()();
    //auto o4 = t()()()(); // Error

    // operator()( size_t )
    t(0)()();
    //t(0)()()(); // Error
    t(0)(0)();
    t(0)()(0);
    const TensorMap<const float,1> i1( t(0)()(1) );
    //i1(3) = 2; // Error
    t(0)(0)(0) = 1;
    //t(0)()(0) = 1; // Error
    Vector<float,2> v = t(0)(0);
    t(0)(0) = v;
    t(0)()(0) = 2*t()(0)(0).topRows(2);
    t(0) += MatrixR<float,2,2>::Zero();
    t(0)(0,0) = 3;
    t(0)(0) = t(0)(0)();

    // operator()( ... )
    t(0,0)();
    t(0,0)(0);
    t(0)(0,0);
    t(0,0,0);
    t()(0,0);

    t.stride(0);
    t.reshape<4>(2,2,2,2)()(1)()(0);

    auto machin = t.reshape<4>(2,2,2,2).contract<1>();
    auto truc = t.contractFirst();
    //Vector<float,8> v2 = t.reshape<3>(4,2,2).contractFirst()()(0);
    //Vector<float,8> v2 = Vector<float, 8>::Ones();
    Vector<float> v2 = Vector<float>::Ones(8);
    MatrixR<float> m3(8,2); m3 << v2, v2;
    //auto bidule = t.reshape<2>(8,2);
    TensorMap<float,2> bidule( m.data(), 8, 2);
    Eigen::Map< MatrixR<float,8,2> >( m.data() ) << v2, v2;
    Eigen::Map< MatrixR<float,8,2> >( bidule.data() ) << v2, v2;
    t.ravel().setZero();

    return 0;
}

template< typename Scalar, size_t dim >
using TensorMap2 = TensorMapTools::TensorMap2<Scalar,dim,0>;

int main( int argc, char** argv )
{
    test_schemes();

    // Const and construction tests
    MatrixR<float,4,4> m = MatrixR<float,4,4>::Constant(42);
    Eigen::Map< const MatrixR<float,4,4> > const_map( m.data() );
    TensorMap2<float,3> t1( m.data(), 4, 2, 2 );
    TensorMap2<float,3> t( m.data(), Eigen::InnerStride<>(m.innerStride()), 4, 2, 2 );
    const TensorMap2<const float,3> t2( m.data(), 4, 2, 2 );
    TensorMap2<const float,3> t3( const_map.data(), Eigen::InnerStride<>(1), 4, 2, 2 );

    //TensorMap2<float,3> t4( const_map.data(), 4, 2, 2 ); // Error
    //TensorMap2<float,3> t5( m, 4, 2, 3 ); // Runtime Error
    //TensorMap2<float,4> t6( m, 4, 2, 2 ); // Error
    //TensorMap2<float,0> t7( m ); // Error
    //TensorMap2<float,2> sub1( t ); // Error

    TensorMap2<float,3> t8 = t;
    TensorMap2<const float,3> aze(t);
    TensorMap2<const float,3> t9 = TensorMap2<float,3>(t);
    TensorMap2<const float,3> t10 = TensorMap2<const float,3>(t);


    // Slicing
    //TensorMap2<float,2> sub2( Slice<3>(0), t );  // Error
    //t.slice<1>( 3 ); // Runtime error
    TensorMap2<float,2> sub3 = t.slice<1>(1);
    TensorMap2<const float,2> sub4 = t.slice<1>(1);
    t2.slice<1>(1);
    TensorMap2<const float,2> sub5 = t2.slice<0>(0);

    // operator()( void )
    auto o1 = t();
    auto o2 = t()();
    auto o3 = t()()();
    //auto o4 = t()()()(); // Error
    TensorMap2<float,3> r1 = o3;
    TensorMap2<const float,3> r2( o3 );
    const TensorMap2<const float,3> r3( o2 );
    o2();
    r3()()();
    // operator()( size_t )
    t(0)()();
    //t(0)()()(); // Error
    t(0)(0)();
    t(0)()(0);
    const TensorMap2<const float,1> i1( t(0)()(1) );
    //i1(3) = 2; // Error
    t(0)(0)(0) = 1;

    //t(0)()(0) = 1; // Error
    Vector<float,2> v = t(0)(0);
    t(0)(0) = v;
    t(0)()(0) = 2*t()(0)(0).topRows(2);
    /*
    t(0) += MatrixR<float,2,2>::Zero();
    t(0)(0,0) = 3;
    t(0)(0) = t(0)(0)();

    // operator()( ... )
    t(0,0)();
    t(0,0)(0);
    t(0)(0,0);
    t(0,0,0);
    t()(0,0);

    t.stride(0);
    t.reshape<4>(2,2,2,2)()(1)()(0);

    auto machin = t.reshape<4>(2,2,2,2).contract<1>();
    auto truc = t.contractFirst();
    //Vector<float,8> v2 = t.reshape<3>(4,2,2).contractFirst()()(0);
    //Vector<float,8> v2 = Vector<float, 8>::Ones();
    Vector<float> v2 = Vector<float>::Ones(8);
    MatrixR<float> m3(8,2); m3 << v2, v2;
    //auto bidule = t.reshape<2>(8,2);
    TensorMap2<float,2> bidule( m.data(), 8, 2);
    Eigen::Map< MatrixR<float,8,2> >( m.data() ) << v2, v2;
    Eigen::Map< MatrixR<float,8,2> >( bidule.data() ) << v2, v2;
    t.ravel().setZero();
     */

    return 0;
}
