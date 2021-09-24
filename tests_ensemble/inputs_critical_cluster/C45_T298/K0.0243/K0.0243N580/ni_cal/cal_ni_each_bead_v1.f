*******************************************************************
*      This Program calculates the interface atoms of proteins
*      attached with other protein
*      It reads two .gro files containing heavy atoms of chain A
*      and chain B respectively.
*      rcut= distance cutoff used for interface calculation
*      nhA = number of heavy atoms in the chain A
*      nhB = number of heavy atoms in the chain B
*      ic_A, ic_B = number of atoms which are part of interface in 
*      chain A and chain B
*******************************************************************      
       program cal_interface
       implicit none
       character(len=50) :: ifl1,ifl2,ofl1,ofl2,ofl3,ofl4
       character(len=8) :: i_char
       integer  i,j,k,l,nframe
       integer  ic_A,ic_B,maxh,maxow,nhA,now
       parameter (maxh=5000,maxow=50000)
       parameter (nframe=1251)
       double precision rdist,dx,dy,dz,rcut
       double precision xA(maxh),yA(maxh),zA(maxh)
       double precision xo(maxow),yo(maxow),zo(maxow)
       character(len=3) :: resname1(maxh),resname2(maxow)
       character(len=7) :: atname1(maxh),atname2(maxow)
       integer  num1(maxh),atnum1(maxh),countow(nframe,maxow)
       integer  num2(maxh),atnum2(maxow)
       integer  sumow(maxow),ind(1,maxh)
       double precision avgow(maxow)
       double precision xbox(nframe),ybox(nframe),zbox(nframe)
       double precision rlow,rhigh
       parameter(rlow=-0.50,rhigh=0.60)

       ifl1='heavyA_wat.gro'
       ifl2='protH.ndx'
       ofl1='ni_surface.dat'
       ofl2='ni_buried.dat'
       ofl3='umb.conf'
       ofl4='surf_atom.gro'
       open(unit=10,status='unknown', file=trim(ifl1))
       open(unit=20,status='unknown', file=trim(ifl2))
       open(unit=30,status='unknown', file=trim(ofl1))
       open(unit=40,status='unknown', file=trim(ofl2))
       open(unit=50,status='unknown', file=trim(ofl3))
       open(unit=60,status='unknown', file=trim(ofl4))
      
***********************************************************************
*          The value of parameters are written here
***********************************************************************
         rcut=0.60d0
         nhA=1598
         now=24048

***********************************************************************
*             Read Coord from Gromacs gro file
***********************************************************************
          read(20,*)
          do i=1,1
           read(20,*)(ind(i,k),k=1,nhA)
          enddo

           countow=0
          do i=1,nframe
          write(*,*)"frame -->",i
          read(10,*)
          read(10,*)
          do j=1,nhA
        read(10,'(i5,1a3,1a7,1i5,3f8.3)')num1(j),resname1(j),atname1(j),
     >    atnum1(j),xA(j),yA(j),zA(j)
           enddo
          do j=1,now
        read(10,'(i5,1a3,1a7,1i5,3f8.3)')num2(j),resname2(j),atname2(j),
     >    atnum2(j),xo(j),yo(j),zo(j)
           enddo
          read(10,'(3f10.5)')xbox(i),ybox(i),zbox(i)
***********************************************************************
*       Estimate Distance between all heavy atoms of Chain A
*       with all waters
***********************************************************************
           do k=1,nhA
            do l=1,now
             dx=xo(l)-xA(k)
             dy=yo(l)-yA(k)
             dz=zo(l)-zA(k)
             dx=dx-dnint(dx/xbox(i))*xbox(i)
             dy=dy-dnint(dy/ybox(i))*ybox(i)
             dz=dz-dnint(dz/zbox(i))*zbox(i)
            rdist=sqrt(dx**2.0d0+dy**2.0d0+dz**2.0d0)

          if(abs(rdist) .le. rcut)then
            countow(i,k)=countow(i,k)+1
           endif
          enddo
         enddo
         enddo

***********************************************************************
*        Save the heavy atoms of Chain A and the number of water
*        within the first solvation shell of each atoms
***********************************************************************
        write(30,'(1a20)')"#Heavy Surface Atoms"
        write(30,'(1a4,2a8,2a6,1a5)')"#num","resname",
     >   "atname","atnum","index","n_i"

        write(40,'(1a19)')"#Heavy Buried Atoms"
        write(40,'(1a4,2a8,2a6,1a5)')"#num","resname",
     >   "atname","atnum","index","n_i"
        write(60,'(1a19)')"Heavy Surface Atoms"

        write(50,'(1A1,1A58)')";","Umbrella potential for ensemble of
     >  spheres exclude water"
        write(50,'(1A1,1A83)')";","Name    Type          Group  Kappa
     > Nstar    mu    width  cutoff  outfile    nstout"

       write(50,'(1A95)')"CAVITATOR dyn_union_sph_sh   OW    0.000
     >   0   0.0    0.01   0.02   nt_ntw_k0_N0.dat   50     \"

***********************************************************************
*        Summing and taking average of number of waters on each H-prot
***********************************************************************
          do i=1,nhA
             sumow=0
             do j=1,nframe
             sumow(i)=sumow(i)+countow(j,i)
             enddo
             avgow(i)=sumow(i)/dble(nframe)
       
           if(avgow(i) .ge. 5.0d0)then
        write(30,'(i5,1a3,1a7,2i6,1f10.3)')num1(i),resname1(i),
     >   atname1(i),atnum1(i),ind(1,i),avgow(i)
       write(60,'(i5,1a3,1a7,1i5,3f8.3)')num1(i),resname1(i),atname1(i),
     >    atnum1(i),xA(i),yA(i),zA(i)

           write(i_char, '(i8)')ind(1,i)
        write(50,'(1f4.1,1f11.2,2a6,1A1)')rlow,rhigh,"",adjustl(i_char),
     > "\"
           elseif(avgow(i) .lt. 5.0d0)then
        write(40,'(i5,1a3,1a7,2i6,1f10.3)')num1(i),resname1(i),
     >   atname1(i),atnum1(i),ind(1,i),avgow(i)
         
           endif
      
           enddo
        write(60,'(3f10.5)')xbox(nframe),ybox(nframe),zbox(nframe)
        write(50,'(1a4,2a10)')";lor","hir","ID"
***********************************************************************
*        Writing Output for number of interfacial atoms
***********************************************************************
c         write(*,'(1a26,2i5)')"#Interface count A & B-->",ic_A,ic_B
*----------------------------------------------------------------------
        write(*,*)
        write(*,*)"*****************************************"
        write(*,'(1a20,1f5.2,1a3)')"Cutoff Distance -->",rcut,"nm"
        write(*,*)
        write(*,*)"*****************************************"
        write(*,*)"Input file used----> ",trim(ifl1)
        write(*,*)"Output file generated----> ",trim(ofl1),", ",
     >trim(ofl2)
       
        stop
        end

