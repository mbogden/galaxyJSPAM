module custom_runs_module

	use init_module
	use parameters_module
	use setup_module
	use io_module
	use integrator
	use df_module

    implicit none

    integer (kind=4) :: pre_setup = 0
    
    contains

    subroutine basic_run(collision_param, npts1, npts2, h1, h2, init_pts, final_pts)
        
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in) :: h1, h2
        
        real(kind=8), intent(out), dimension(npts1+npts2+1,6) :: init_pts, final_pts

        real (kind=8) :: t0, time_interval
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr
        integer :: nparticles1, nparticles2

        ! print *, "FR: Basic Run!"

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call my_init()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        call SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, h1, h2)
    
        ! print *, "FR: Creating disk!"
        call CREATE_COLLISION
            
        ! ! Saving initial particles
        ! copy the initial particles
        init_pts = x0

        ! initialize rk routine for particle integration/perturbation
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        t0 = tstart

        nstep = int( (tend - t0) / h) + 2
        nstep_local = nstep 

        time_interval = (tend - t0) * 2

        nunit = 50

        ! This writes parameters to disk "fort.50"
        ! call OCTAVE_PARAMETERS_OUT(mass1, theta1, phi1, rout1, mass2, &
        !     theta2, phi2, rout2, original_rv(1:3), original_rv(4:6), time_interval, x0(n,:), n, nunit)

        ! main integration loop
        iout = 0
        do istep = 1, nstep_local
            call TAKE_A_STEP
            rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

            if (mod(istep, showsteps) == 0) then
        !       print*,"q",istep, time, rrr
            endif

            if (mod(istep, 50) == 5 .and. show_all_steps) then
                call CREATE_IMAGES
            endif
        enddo

        ! Save final pts
        final_pts = x0

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call more_cleanup()

        return
        
    end subroutine basic_run

    subroutine orbit_run(collision_param, in_n_steps, orbit_path)
        
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(in) :: in_n_steps
        real(kind=8), intent(out), dimension(in_n_steps,6) :: orbit_path

        real (kind=8) :: t0, time_interval
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr
        integer :: nparticles1, nparticles2

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call my_init()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        call SETUP_CUSTOM_COLLISION(collision_param, 10, 5, 0.0d0, 0.0d0)
    
        ! print *, "FR: Creating disk!"
        call CREATE_COLLISION
            
        ! initialize rk routine for particle integration/perturbation
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        t0 = tstart
        nstep = int( (tend - t0) / h) + 2
        nstep_local = nstep 
        nunit = 50

        ! main integration loop
        iout = 0

        do istep = 1, nstep_local
            call TAKE_A_STEP
            rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

            orbit_path(istep,:) = x0(n,:)

        enddo

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call more_cleanup()

        return
        
    end subroutine orbit_run

    subroutine calc_nsteps( collision_param, out_n_steps )
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(out) :: out_n_steps

        real (kind=8) :: t0, time_interval
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call my_init()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        call SETUP_CUSTOM_COLLISION(collision_param, 10, 5, 0.0d0, 0.0d0)
    
        ! print *, "FR: Creating disk!"
        call CREATE_COLLISION
            
        ! initialize rk routine for particle integration/perturbation
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        t0 = tstart

        nstep = int( (tend - t0) / h) + 2

        out_n_steps = nstep
        
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call more_cleanup()

        return

    end subroutine calc_nsteps


    subroutine basic_disk(collision_param, npts1, npts2, h1, h2, init_pts)
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in) :: h1, h2
        real(kind=8), intent(in), dimension(22) :: collision_param
        real(kind=8), intent(out), dimension(npts1+npts2+1,6) :: init_pts

        ! print *, "FR: basic_disk!"

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call my_init()
            pre_setup = 1
        endif

        ! Define collision parameters
        call SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, h1, h2)
    
        ! print *, "FR: Creating disk!"
        call CREATE_COLLISION
            
        ! ! Output particles to disk
        ! print *, "FR: Returning particles to disk!"
        init_pts = x0

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        return
        
    end subroutine basic_disk


    subroutine SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, h1, h2)

        implicit none

        ! Variable declarations
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in) :: h1, h2
        real(kind=8), intent(in), dimension(22) :: collision_param

        ! Just saying hi!
        ! print *, "FR: SETUP_CUSTOM_COLLISION!"

        ! Setup custom collision
        ! ! call CREATE_COLLISION
        ! ! call INIT_DISTRIBUTION
      
        ! print
        ! print *, "FR: Setting up custom collision"

        ! ! set the collision parameters
        potential_type = 0
        sec_vec(1) = collision_param(1)
        sec_vec(2) = collision_param(2)
        sec_vec(3) = collision_param(3)
        sec_vec(4) = collision_param(4)
        sec_vec(5) = collision_param(5)
        sec_vec(6) = collision_param(6)
        mass1 = collision_param(7)
        mass2 = collision_param(8)
        rout1 = collision_param(9)
        rout2 = collision_param(10)
        phi1 = collision_param(11)
        phi2 = collision_param(12)
        theta1 = collision_param(13)
        theta2 = collision_param(14)
        epsilon1 = collision_param(15)
        epsilon2 = collision_param(16)
        rscale1(1) = collision_param(17)
        rscale1(2) = collision_param(18)
        rscale1(3) = collision_param(19)
        rscale2(1) = collision_param(20)
        rscale2(2) = collision_param(21)
        rscale2(3) = collision_param(22)
        use_sec_vec = .true. 

        ! ! Derived values
        eps1 = epsilon1 * epsilon1
        eps2 = epsilon2 * epsilon2

        ! Define simulation parameters
        n1 = npts1
        n2 = npts2
        n = n1 + n2

        ! Allocate the space for the particles - n+1 here ONLY
        allocate(x0(n+1,6), stat=iostat)
        allocate(xout(n+1,6), stat=iostat)
        allocate(projected(n+1,3), stat=iostat)
        ! print *, "FR: n1, n2, n", n1, n2, n
        ! print *, "FR: Points allocated!"

        heat1 = h1
        heat2 = h2

        ! WORKING comment to see if this fixes pts offset
        ! Well, it didn't fix it, nor did it break anything.  Leave commented? 
        ! Nope, something definitely broke.  Uncommenting
        tIsSet = .false.
        tStart = -5
        time = tStart

        ! No integration backwards? 
        if ( tStart == 0 ) then
            time = 0.0d0
            tIsSet = .true.

        else
            time = tStart
            tIsSet = .false.
        endif
    
            
    ! set the default collision  - testing only

    end subroutine SETUP_CUSTOM_COLLISION


    subroutine more_cleanup()
    
        ! allocate(ival11(n))
        ! allocate(ival22(n))
        ! allocate(ivaln(n))
        ! allocate(df_force11(n))
        ! allocate(df_force22(n))
        ! allocate(df_forcen(n))
        ! allocate(c3n(n))

        deallocate(ival11)
        deallocate(ival22)
        deallocate(ivaln)
        deallocate(df_force11)
        deallocate(df_force22)
        deallocate(df_forcen)
        deallocate(c3n)

    end subroutine more_cleanup

    subroutine my_init()

        implicit none

        integer :: nparticles1, nparticles2

        ! Setup default parameters for a quick test
        nparticles1 = 100
        nparticles2 = 50
        call RANDOM_SEED()
        call STANDARD_GALAXY1(mass1, epsilon1, rin1, rout1, rscale1, theta1, phi1, opt1, heat1 )
        call STANDARD_GALAXY2(mass2, epsilon2, rin2, rout2, rscale2, theta2, phi2, opt2, heat2)
        
        n1 = nparticles1
        n2 = nparticles2
        n = n1 + n2 

        call TEST_COLLISION(n, n1, n2, time, inclination_degree, omega_degree, &
          rmin, velocity_factor, h, nstep, nout) 

    end subroutine my_init

    

    subroutine s(n, m, c, x)
        integer, intent(in) :: n, m
        real(kind=8), intent(out), dimension(n, m) :: x
        real(kind=8), intent(in) :: c(:)

        x = 0.0d0
        if (size(c) > 0) then
            x(1, 1) = c(1)
        endif
    end subroutine s

    subroutine print_message(message)
        implicit none
        character(len=*), intent(in) :: message
        print *, "From Fortran: ", message
    end subroutine print_message

end module custom_runs_module
