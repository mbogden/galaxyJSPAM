program basic_run



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
!------------------------------------------------------
!
!
!

! set the disk parameters
  call RANDOM_SEED()


! set the target parameters
! set the target parameters
  nparticles1 = 1000!0!100000!10000
  nparticles2 = 500!0!50000!5000
  call DEFAULT_PARAMETERS(nparticles1, nparticles2)


  call CREATE_COLLISION

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

!      call CREATE_IMAGES

  fname = trim(outfilename) // "_" // trim(distinguisher) // ".000"
  open(unit, file=trim(fname))
  call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
       eps1, eps2, &
       n, n1, n2, &
       time, header_on)
  close(unit)


! main integration loop
  iout = 0
  do istep = 1, nstep_local
    call TAKE_A_STEP
    rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

    if (mod(istep, showsteps) == 0) then
       print*,"q",istep, time, rrr
    endif



    if (mod(istep, 50) == 5 .and. show_all_steps) then
        call CREATE_IMAGES
    endif
  enddo

!      call CREATE_IMAGES
  fname = trim(outfilename) // "_" // trim(distinguisher) // ".101"
  open(unit, file=trim(fname))
  call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
       eps1, eps2, &
       n, n1, n2, &
       time, header_on)
  close(unit)

! this creates a simple script for animating the output with gnuplot
! gnuplot
! i = 1
! j = 2
! load 'gscript
  if (.not. header_on) then
    call CREATE_GNUPLOT_SCRIPT(x0, iout)
  endif

! clean up memory
  deallocate(x0)
  deallocate(xout)
  call DEALLOCATE_RKVAR


! enddo


end program basic_run


