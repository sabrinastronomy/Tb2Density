module zreion_tools
  ! Intel
  use IFPORT
  use OMP_LIB


  ! Global
  use zreion_global


  ! Tools
  use fft_tools


  ! Default
  implicit none


contains


  subroutine init_arrays
    ! Default
    implicit none


    ! Local variables
    integer(4) :: k


    ! Timing variables
    character(8) :: ts1,ts2
    real(8)      :: tr1,tr2
    call time(ts1)
    tr1 = omp_get_wtime()


    ! Allocate arrays
    allocate(zreion( N_grid+2,N_grid,N_grid))
    allocate(density(N_grid  ,N_grid,N_grid))

    ! First touch in parallel
    !$omp parallel   &
    !$omp default(shared) &
    !$omp private(k)
    !$omp do
    do k=1,N_grid
       zreion(:,:,k) = 0
    enddo
    !$omp end do
    !$omp do
    do k=1,N_grid
       density(:,:,k) = 0
    enddo
    !$omp end do
    !$omp end parallel


    tr2 = omp_get_wtime()
    call time(ts2)
    write(*,'(f8.2,2a10,a)') tr2-tr1,ts1,ts2,'  Called init arrays'
    return
  end subroutine init_arrays


!------------------------------------------------------------------------------!


  subroutine calc_zreion
    ! Default
    implicit none


    ! Local parameters
    real(8), parameter :: Ak = 2*pi/ngridr


    ! Local variables
    integer(4) :: i,j,k
    real(8)    :: kr,kx,ky,kz
    real(8)    :: wcell,wsmooth,x
    real(8)    :: z,zavg,zsig,zmax,zmin


    ! Timing variables
    character(8) :: ts1,ts2
    real(8)      :: tr1,tr2
    call time(ts1)
    tr1 = omp_get_wtime()


    ! Move density field to zreion grid
    !$omp parallel do &
    !$omp private(i,j,k)
    do k=1,N_grid
       do j=1,N_grid
          do i=1,N_grid
             zreion(i,j,k) = density(i,j,k) - 1 ! converting from density to overdensity
          enddo
       enddo
    enddo
    !$omp end parallel do

    ! Forward FFT of field
    call fft_3d('f',N_grid,N_grid,N_grid,zreion)

    ! Calculate zreion field in Fourier space
    !$omp parallel do              &
    !$omp default(shared)          &
    !$omp private(i,j,k)           &
    !$omp private(kx,ky,kz,kr)     &
    !$omp private(x,wcell,wsmooth)
    do k=1,N_grid
       if (k <= N_grid/2+1) then
          kz = Ak*(k-1)
       else
          kz = Ak*(k-1-N_grid)
       endif

       do j=1,N_grid
          if (j <= N_grid/2+1) then
             ky = Ak*(j-1)
          else
             ky = Ak*(j-1-N_grid)
          endif

          do i=1,N_grid+2,2
             kx = Ak*((i-1)/2)

             kr = sqrt(kx**2 + ky**2 + kz**2)*ngridr/box

             ! Deconvolve cell, not needed when not using particles
             wcell = (sinc(kx/2)*sinc(ky/2)*sinc(kz/2))**2

             ! Smooth with top hat window
             x       = kr*Rsmooth
             wsmooth = tophat(x)

             ! Apply bias relation
             zreion(i:i+1,j,k) = zreion(i:i+1,j,k)*bias(kr)*wsmooth/wcell
          enddo
       enddo
    enddo
    !$omp end parallel do

    ! Inverse FFT zreion field
    call fft_3d('b',N_grid,N_grid,N_grid,zreion)

    ! Calculate zreion field
    zavg = 0
    zsig = 0
    zmax = 0
    zmin = huge(zmin)

    !$omp parallel do &
    !$omp default(shared) &
    !$omp private(i,j,k,z) &
    !$omp reduction(+:zavg,zsig) &
    !$omp reduction(max:zmax) &
    !$omp reduction(min:zmin)
    do k=1,N_grid
       do j=1,N_grid
          do i=1,N_grid
             z = zmean_zre + (1+zmean_zre)*zreion(i,j,k)
             zreion(i,j,k) = z

             zavg = zavg + z
             zsig = zsig + (z-zmean_zre)**2
             zmax = max(zmax,z)
             zmin = min(zmin,z)
          enddo
       enddo
    enddo
    !$omp end parallel do

    ! Write to standard output
    zavg = zavg/ncellr
    zsig = sqrt(zsig/ncellr)
    write(*,*) "zreion: ",real((/zavg,zsig,zmax,zmin/))


    tr2 = omp_get_wtime()
    call time(ts2)
    write(*,'(f8.2,2a10,a)') tr2-tr1,ts1,ts2,'  Called calc zreion'
    return


  contains


    pure function bias(k)
      ! Default
      implicit none

      ! Function arguments
      real(8), intent(in) :: k
      real(8)             :: bias

      bias = b0_zre/(1 + k/kb_zre)**alpha_zre

      return
    end function bias


    pure function sinc(x)
      ! Function arguments
      real(8), intent(in) :: x
      real(8)             :: sinc

      if (abs(x) > 1D-6) then
         sinc = sin(x)/x
      else
         sinc = 1 - x**2/6
      endif

      return
    end function sinc


    pure function tophat(x) ! this is called in fourier space, so x will = k
      ! Function arguments
      real(8), intent(in) :: x
      real(8)             :: tophat

      if (abs(x) > 1D-6) then
         tophat = 3*(sin(x)-cos(x)*x)/x**3 ! 3D FT of sphere, smoothes by averaging things smaller than sphere
      else
         tophat = 1 - x**2/10 ! taylor expansion
      endif

      return
    end function tophat


  end subroutine calc_zreion


end module zreion_tools
