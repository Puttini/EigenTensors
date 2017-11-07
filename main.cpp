#include <iostream>
#include "ConstScheme1.h"
#include "ConstScheme2.h"

template< typename Scalar >
using ConstScheme = ConstScheme2<Scalar>;

void f1( const int& i )
{}
void f2( int i )
{}
void f3( int& i )
{}

int main( int argc, char** argv )
{
    // const tests
    int i = 0;
    ConstScheme<int> a(&i);
    ConstScheme<const int> b(&i);
    ConstScheme<int> c(a);
    ConstScheme<const int> d(b);
    ConstScheme<const int> e(a);

    f1( a.f() );
    f1( b.f() );
    f2( a.f() );
    f2( b.f() );
    f3( a.f() );

    // Inheritance test ?
    return 0;
}
