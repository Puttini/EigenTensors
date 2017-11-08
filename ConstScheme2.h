#ifndef EIGENTENSORS_CONSTSCHEME2_H
#define EIGENTENSORS_CONSTSCHEME2_H

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

// -----------------------------------------------------------------------------------

template< typename Derived, typename Scalar >
struct Base
{
    Scalar* i;

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    Base( Scalar* i ) : i(i)
    {}
    Base( Derived& o ) : i(o.i)
    {}
    Base( NonConst<Derived>&& o ) : i(o.i)
    {}

    Scalar f() const
    {
        std::cout << "const" << std::endl;
        return *i;
    }
    Scalar& f()
    {
        std::cout << "non-const" << std::endl;
        return *i;
    }

    Derived copy()
    { return Derived( derived() ); }
    Derived copy() const
    { return Derived( derived() ); }
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

// -----------------------------------------------------------------------------------

#define BASE_OF( Derived, Scalar ) Base< ConstAs<Scalar,Derived<NonConst<Scalar>>>, Scalar >

template< typename Scalar >
struct ConstScheme3 : public BASE_OF( ConstScheme3, Scalar )
{
    typedef BASE_OF( ConstScheme3, Scalar ) MyBase;
    using MyBase::Base;
};

#endif //EIGENTENSORS_CONSTSCHEME2_H
