program mod_run


  use init_module
  use parameters_module
  use setup_module
  use io_module
  use integrator


  implicit none

  character (len=20):: target_name
  real (kind=8) :: t0, time_interval
  integer (kind=4) :: nstep_local
  real (kind=8) :: rrr
  integer :: nparticles1, nparticles2


  real (kind=8) :: ttmin, rxmin, rrxmin(6)

!------------------------------------------------------
!
!
!

! set the disk parameters
  call RANDOM_SEED()

! set the target parameters
  nparticles1 = 1
  nparticles2 = 1
  call DEFAULT_PARAMETERS(nparticles1, nparticles2)

  call CREATE_COLLISION()



!
!     -----loop over the system for the output
!

! initialize rk routine
  call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
       theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)


  t0 = tstart

  nstep = int( (tend - t0) / h) + 2
  nstep_local = nstep 

  time_interval = (tend - t0) * 2

  nunit = 50
  call OCTAVE_PARAMETERS_OUT(mass1, theta1, phi1, rout1, mass2, &
       theta2, phi2, rout2, original_rv(1:3), original_rv(4:6), time_interval, x0(n,:), n, nunit)


! main integration loop
  rxmin = 100000.
  rrxmin = 0
  ttmin = -1000
  do istep = 1, nstep_local
    call TAKE_A_STEP
    rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))
    if (rrr < rxmin) then
       rxmin = rrr
       rrxmin = x0(n,:)
       ttmin = istep
    endif
  enddo


  print*, (tend-t0)-ttmin*h, rxmin, rrxmin

! clean up memory
  deallocate(x0)
  deallocate(xout)
  call DEALLOCATE_RKVAR


! enddo


end program mod_run


