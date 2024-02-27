# Smooth Particle Hydrodynamics (SPH) for Interactive Application

**SPH** is an interpolation method for particle systems. With SPH, field quantities that are only defined at discrete particle locations can be evaluated anywhere in space. For this purpose, SPH distributes quantities in a local neighborhood of each particle using *radial symmetrical smoothing kernels*.

According to SPH, a scalar quantity A is interpolated at location r by a weighted sum of contributions from all particles: $$ f(r) = \sum_{j\not={i}} m_j \frac{f_j}{\rho_j} W(r-r_j, h) \tag{1}$$ where j iterates over all particles, $m_j$ is the mass of particle $j$, $r_j$ its position, $ρ_j$ the density and A j the field quantity at $r_j$.

The function $W(r,h)$ is called the **smoothing kernel** with core radius $h$. $w$ is even $(i.e. W(r,h)=W(-r,h))$ and normalized $(\int W(r) dr = 1)$, and it is of second order accuracy.
The update fro densities is performed using Eqn. 1: $$ \rho(r) = \sum_{j\not={i}} m_j \frac{\rho_j}{\rho_j} W(r-r_j, h) = \sum_{j\not={i}} m_j W(r-r_j, h) \tag{2} $$ In most fluid equations, derivatives of field quantities appear and need to be evaluated. With the SPH approach, such derivatives only affect the smoothing kernel. The gradient of A is simply: $$ \nabla f(r) = \sum_{j\not={i}} m_j \frac{f_j}{\rho_j} \nabla W(r-r_j, h) \tag{3}$$ while the Laplacian of A evaluates to $$ \nabla^2 f(r) = \sum_{j\not={i}} m_j \frac{f_j}{\rho_j} \nabla^2 W(r-r_j, h) \tag{4}$$ It is important to realize that SPH holds some inherent problems. When using SPH to derive fluid equations for particles, these equations are not guaranteed to satisfy certain physical principals such as symmetry of forces and conservation of momentum. The next section describes our SPH-based model and techniques to solve these SPH-related problems.

## Modelling Fluid with Particles
In the Eulerian (grid based) formulation, isothermal fluids are described by a velocity field $v$, a density field $\rho$ and a pressure field $p$. The evolution of these quantities over time is given by two equations. The first equation assures conservation of mass:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (pv) = 0 \tag{5}$$
while the Navier-Stokes equation for incompressible fluids formulates conservation of momentum:
$$ \rho (\frac{\partial v}{\partial t} + v \cdot \nabla v ) = -\nabla p + \rho g + \mu \nabla^2 v \tag{6}$$
The use of particles instead of a stationary grid simplifies these two equations substantially. First, because the number of particles is constant and each particle has a constant mass, mass conservation is guaranteed and Eqn. (5) can be omitted completely. Second, the expression $(\frac{\partial v}{\partial t} + v \cdot \nabla v )$ on the left hand side of Eqn. (6) can be replaced by the substantial derivative $\frac{Dv}{Dt}$. Since the particles move with the fluid, the substantial derivative of the velocity field is simply the time derivative of the velocity of the particles meaning that the convective term $v\cdot \nabla v$ is not needed for particle systems.

There are three force density fields left on the right hand side of Eqn. (6) modeling pressure $(−\nabla p)$, external forces $(\rho g)$ and viscosity $(\mu \nabla^2 v)$. The sum of these force density fields $ f = -\nabla p + \rho g + \mu \nabla^2 v $ determines the change of momentum $\rho \frac{Dv}{Dt}$ of the particles on the left hand side. For the acceleration of particle i we, thus, get:
$$ a_i = \frac{dv_i}{dt} = \frac{f_i}{\rho_i} \tag{7}$$

### Pressure
Application of the SPH rule described in Eqn. (3) to the pressure term $(−\nabla p)$ yields: $$ f_{i}^{pressure} = -\nabla p(r_i) = - \sum_{j\not={i}} m_j \frac{p_j}{\rho_j} \nabla W(r_i-r_j, h) \tag{8}$$ Unfortunately, this force is not symmetric as can be seen when only two particles interact. Since the gradient of the kernel is zero at its center, particle i only uses the pressure of particle j to compute its pressure force and vice versa. Because the pressures at the locations of the two particles are not equal in general, the pressure forces will not be symmetric. Different ways of symmetrization of Eqn. (8) have been proposed in the literature. Taking average of the pressures is best suited for our purposes of speed and stability. $$ f_{i}^{pressure} = - \sum_{j\not={i}} m_j \frac{p_i+p_j}{2\rho_j} \nabla W(r_i-r_j, h) \tag{9}$$ Since particles only carry the three quantities mass, position and velocity, the pressure at particle locations has to be evaluated first. This is done in two steps. Eqn. (2) yields the density at the location of the particle. Then, the pressure can be computed via the ideal gas state equation $$ p = k \rho, \tag{10}$$ where k is a gas constant that depends on the temperature. In our simulations we use a modified version of Eqn. (11) suggested by Desbrun $$ p = k (\rho - \rho_0), \tag{11}$$ where $\rho_0$ is the rest density. Since pressure forces depend on the gradient of the pressure field, the offset mathematically has not effect on pressure forces. However, the offset does influence the gradient of a field smoothed by SPH and makes the simulation numerically more stable.

### Viscosity
Application of the SPH rule to the viscosity term $(\mu \nabla^2 v)$ again yields asymmetric forces $$ f_{i}^{viscosity} = \mu \nabla^2 v_i = \mu \sum_{j\not={i}} m_j \frac{v_j}{\rho_j} \nabla^2 W(r_i-r_j, h) \tag{12}$$ because the velocity field varies from particle to particle. Since viscosity forces are only dependent on velocity differences and not on absolute velocities, there is a natural way to symmetrize the viscosity forces by using velocity differences: $$ f_{i}^{viscosity} = \mu \sum_{j\not={i}} m_j \frac{v_j-v_i}{\rho_j} \nabla^2 W(r_i-r_j, h) \tag{13}$$

### Surface Tension
We model surface tension forces (not present in Eqn. (6)) explicitly based on ideas of Morris. Molecules in a fluid are subject to attractive forces from neighboring molecules. Inside the fluid these intermolecular forces are equal in all directions and balance each other. In contrast, the forces acting on molecules at the free surface are unbalanced. The net forces (i.e. surface tension forces) act in the direction of the surface normal towards the fluid. They also tend to minimize the curvature of the surface. The larger the curvature, the higher the force. Surface tension also depends on a tension coefficient σ which depends on the two fluids that form the surface.

The surface of the fluid can be found by using an additional field quantity which is 1 at particle locations and 0 everywhere else. This field is called **color field** in the literature. For the smoothed color field we get: $$ C_i = \sum_{j\not={i}} m_j \frac{1}{\rho_j} \nabla^2 W(r_i-r_j, h) \tag{14} $$ The gradient field of the smoothed color field $$ n = \nabla C \tag{15}$$ yields the surface normal field pointing into the fluid and the divergence of n measures the curvature of the surface $$ \kappa = \frac{-\nabla^2C}{||n||} \tag{16}$$ The minus is necessary to get positive curvature for convex fluid volumes. Putting it all together, we get for the surface traction $$ t^{surface} = \sigma \kappa \frac{n}{||n||} \tag{17}$$ To distribute the surface traction among particles near the surface and to get a force density we multiply by a normalized scalar field $\delta s = ||n||$ which is non-zero only near the surface. For the force density acting near the surface we get $$ t^{surface} = \sigma \kappa n = -\sigma \nabla^2 C \frac{n}{||n||} \tag{18}$$ Evaluating $\frac{n}{||n||}$ at locations where $||n||$ is small causes numerical problems. We only evaluate the force if $||n||$ exceeds a certain threshold.

### Smoothing Kernels
Stability, accuracy and speed of the SPH method highly depend on the choice of the smoothing kernels. The kernels we use have second order interpolation errors because they are all even and normalized. In addition, kernels that are zero with vanishing derivatives at the boundary are conducive to stability.
$$ W_{poly6}(r,h) = \frac{315}{64\pi h^9} \begin{cases}
      (h^2-r^2)^3 & 0 \le r \le h\\
      0 & \text{otherwise}
    \end{cases}    \tag{19}  $$ 
$$ \nabla W_{poly6}(r,h) = \frac{945}{32\pi h^9} \begin{cases}
      r(h^2-r^2)^2 & 0 \le r \le h\\
      0 & \text{otherwise}
    \end{cases}    \tag{20}  $$This is used in all but two cases. 
If this kernel is used for the computation of the pressure forces, particles tend to build clusters under high pressure. As particles get very close to each other, the repulsion force vanishes because the gradient of the kernel approaches zero at the center. Desbrun solves this problem by using a spiky kernel with a non vanishing gradient near the center. For pressure computations, Debrun’s spiky kernel 
$$ W_{spiky}(r,h) = \frac{15}{\pi h^6} \begin{cases}
      (h-r)^3 & 0 \le r \le h\\
      0 & \text{otherwise}
    \end{cases}    \tag{21}  $$ 
$$ \nabla W_{spiky}(r,h) = \frac{45}{\pi h^6} \begin{cases}
      (h-r)^2 & 0 \le r \le h\\
      0 & \text{otherwise}
    \end{cases}    \tag{22}  $$generates the necessary repulsion forces. At the boundary where it vanishes it also has zero first and second derivatives.
Viscosity is a phenomenon that is caused by friction and, thus, decreases the fluid’s kinetic energy by converting it into heat. Therefore, viscosity should only have a smoothing effect on the velocity field. However, if a standard kernel is used for viscosity, the resulting viscosity forces do not always have this property. For two particles that get close to each other, the Laplacian of the smoothed velocity field (on which viscosity forces depend) can get negative resulting in forces that increase their relative velocity. The artifact appears in coarsely sampled velocity fields. In real-time applications where the number of particles is relatively low, this effect can cause stability problems. For the computation of viscosity forces, a third kernel: 
$$ W_{viscosity}(r,h) = \frac{15}{2\pi h^3} \begin{cases}
      -\frac{r^3}{2h^3}+\frac{r^2}{h^2}+\frac{h}{2r}-1 & 0 \le r \le h \\
      0 & \text{otherwise}
\end{cases}    \tag{23}  $$ whose Laplacian, $$ \nabla^2W(r,h) = \frac{45}{\pi h^6} (h-r) \tag{24} $$ is positive everywhere with the following additional properties: $$ W(||r||=h, h) = \nabla W(||r||=h, h) = 0 $$ The use of this kernel for viscosity computations increased the stability of the simulation significantly allowing to omit any kind of additional damping.