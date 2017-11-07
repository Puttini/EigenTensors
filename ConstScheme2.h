#ifndef EIGENTENSORS_CONSTSCHEME2_H
#define EIGENTENSORS_CONSTSCHEME2_H

template< typename Derived, typename Scalar >
struct Base
{
    Scalar* i;

    Base( Scalar* i ) : i(i)
    {}

    Base( Derived& o ) : i(o.i)
    {}

    Scalar f() const { return *i; }
    Scalar& f() { return *i; }
};

template< typename Scalar >
struct ConstScheme2 : public Base< ConstScheme2<Scalar>, Scalar >
{
    using Base< ConstScheme2<Scalar>, Scalar >::Base;
};

template< typename Scalar >
struct ConstScheme2< const Scalar > : public Base< const ConstScheme2<Scalar>, const Scalar >
{
    using Base< const ConstScheme2<Scalar>, const Scalar >::Base;
};

#endif //EIGENTENSORS_CONSTSCHEME2_H
