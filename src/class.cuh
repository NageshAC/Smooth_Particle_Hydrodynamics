#pragma once
#include <iostream>
#include <iomanip>

#define _USE_MATH_DEFINES
#include <math.h>

template <class Typ>
class Coordinates {
    public:
    Typ x{}, y{}, z{};
};



template <class Typ>
class Rel_Coordinates : public Coordinates<Typ> {
    public:
    // unsigned idx_a{}, idx_b{};
    Typ dist{}, dist_inv{};

};


template <class Typ>
class Rel_Force_Vector : public Rel_Coordinates<Typ> {
public:
    Coordinates<Typ>    W_poly6{}, 
                        lap_W_poly6{}, 
                        grad_W_spiky{}, 
                        lap_W_vis{};
};

template <class Typ>
class Tensors : public Coordinates<Typ> {
    Coordinates<Typ> velocity{}, force {};
};