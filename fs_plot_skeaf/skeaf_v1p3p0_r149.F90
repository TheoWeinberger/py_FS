PROGRAM skeaf

!----------------------------------------------------------------------------------------
! Supercell K-space Extremal Area Finder (SKEAF) version 1.3.0 release build 149
!
! Copyright (C) 2012 Patrick Rourke (rourke@gmail.com)
!
! This program is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation, either version 3 of the License, or (at your option) any later version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT ANY
! WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
! PARTICULAR PURPOSE.  See the GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along with this
! program (default filename is COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
!
! Please cite P.M.C. Rourke and S.R. Julian, Comput. Phys. Commun. 183, 324 (2012)
! in any publications making use of SKEAF.
!----------------------------------------------------------------------------------------

IMPLICIT NONE

integer, parameter :: int4byte = SELECTED_INT_KIND(9) ! for defining 4-byte integers
integer, parameter :: int8byte = SELECTED_INT_KIND(18) ! for defining 8-byte integers

! IMPORTANT: set the parameter maxnumint to an integer value sustainable by your computer/RAM set-up.
! The conservative default that should work on most systems 150.
! If you experience segmentation faults or killed processes, try reducing this value
!   (conversely, many systems can support a maxnumint value greater than 150).
! The bundled program "memoryestimateskeaf_for_v1p3p0.F90" can provide a rough estimate for the
!   maximum maxnumint supported by a given amount of free memory.
! Note that in order to utilize more than 2GB of memory, a 64-bit system is required and compiler 
!   options "ifort -mcmodel=medium -shared-intel" must be added.
integer(KIND=int8byte), parameter :: maxnumint = 150 ! Maximum number of points per single cell side (a full supercell side will have 4x this many)

! The following are variables derived from maxnumint, which are used in the dimensioning of arrays
integer(KIND=int8byte), parameter :: scarraydimension = (4*maxnumint)+2 ! Originally 602, ie. when maxnumint is 150
integer(KIND=int8byte), parameter :: slicedimension = (4*maxnumint)*(4*maxnumint) ! Originally 360000, ie. when maxnumint is 150
integer(KIND=int8byte), parameter :: measpardim = (30*maxnumint)+300 ! Originally 4800, ie. when maxnumint is 150
integer(KIND=int8byte), parameter :: measpardim2 = 4*measpardim ! Originally 19200, ie. when maxnumint is 150

! Other misc hard-coded parameters
character(len=11), parameter :: buildnumber = 'v1.3.0 r149'
character(len=12), parameter :: releasedate = '29 Jul. 2012'
integer(KIND=int8byte), parameter :: maxnuminput = 100 ! Maximum number of points per side in the input file RUC; don't change this unless your input BXSF file has nx, ny or nz > 100
integer(KIND=int8byte), parameter :: maxinputarraysize = maxnuminput*maxnuminput*maxnuminput

! Numeric parameter declarations
double precision, parameter :: pi = 3.1415926535897932D0
double precision, parameter :: convau2ang = 0.529177209D0 ! Note: this is the conversion from atomic units (Bohr) to angstroms, because BXSF file reciprocal lattice vectors are given in (atomic units)^-1
double precision, parameter :: convfsarea2kt = 10.47576797D0 ! Note: this multiplicative constant is h_bar/(2*pi*elementarycharge) in units so that F ends up in kT (for fsarea given in angstroms^-2)
double precision, parameter :: convfsdade2mstar = 0.089135845D0 ! Note: this multiplicative constant is (h_bar^2)/(2*pi*(m_e)) in units so that m* is dimensionless (for fsdAde given in (angstroms^-2)*Ryd^-1)

! Variable declarations
character(len=100),save::parsread
character(len=50),save::filename
character(len=100),save::tempstringread
logical,save::badinput
logical,save::runonce
double precision,save::readfermienergy
double precision,save::fermienergy
double precision,save::numlistedbands
integer(KIND=int8byte),save::nx
integer(KIND=int8byte),save::ny
integer(KIND=int8byte),save::nz
integer(KIND=int8byte),save::numkpoints
integer(KIND=int8byte),save::numread
logical,save::keepreading
double precision,save::lreciplatx
double precision,save::lreciplaty
double precision,save::lreciplatz
double precision,save::maxlreciplat
double precision,save::patricklreciplatx
double precision,save::patricklreciplaty
double precision,save::patricklreciplatz
double precision,save::patricklreciplato1
double precision,save::patricklreciplato2
double precision,save::patricklreciplato3
double precision,save::patricklreciplatx1
double precision,save::patricklreciplatx2
double precision,save::patricklreciplatx3
double precision,save::patricklreciplaty1
double precision,save::patricklreciplaty2
double precision,save::patricklreciplaty3
double precision,save::patricklreciplatz1
double precision,save::patricklreciplatz2
double precision,save::patricklreciplatz3
double precision,save::plrx1
double precision,save::plrx2
double precision,save::plrx3
double precision,save::plry1
double precision,save::plry2
double precision,save::plry3
double precision,save::plrz1
double precision,save::plrz2
double precision,save::plrz3
double precision,save::kpointspacingx
double precision,save::kpointspacingy
double precision,save::kpointspacingz
double precision,save::minkpointspacing
double precision,save::anglelatxy
double precision,save::anglelatxz
double precision,save::anglelatyz
double precision,save::freq2kxy
double precision,save::freq2kxz
double precision,save::freq2kyz
double precision,save::freq2kmax
double precision,dimension(maxinputarraysize),save::kreadarray
double precision,dimension(maxnuminput,maxnuminput,maxnuminput),save::masterkarray
character(len=scarraydimension),dimension(scarraydimension,scarraydimension),save::slicepic
character(len=scarraydimension),save::slicescale
integer(KIND=int8byte),save::numintkpoints
double precision,save::intkpointspacingx
double precision,save::intkpointspacingy
double precision,save::intkpointspacing
character(len=100),save::intcorrect
character(len=100),save::hvd
character(len=100),save::allowextnearwalls
integer(KIND=int8byte),save::numslices
integer(KIND=int8byte),save::numx
integer(KIND=int8byte),save::numy
character(len=7),save::slicelinefmt
double precision,save::xkseparation
double precision,save::ykseparation
double precision,save::xlength
double precision,save::ylength
double precision,save::minarea
character(len=100),save::parcorrect

double precision,save::rotelectronarea
double precision,save::rotholearea
double precision,save::percentelectronlike
double precision,save::percentholelike
double precision,save::bandvolume
double precision,save::bandholevolume
double precision,save::bandnumelectrons
double precision,save::bandnumholes

integer(KIND=int8byte),dimension(scarraydimension),save::numslfs
integer(KIND=int8byte),dimension(scarraydimension),save::numslopen
integer(KIND=int8byte),dimension(scarraydimension),save::numsltoosmall
double precision,dimension(scarraydimension,measpardim),save::slfsorbtype
integer(KIND=int8byte),dimension(scarraydimension,measpardim),save::slfsorbnum
double precision,dimension(scarraydimension,measpardim),save::slfsarea
double precision,dimension(scarraydimension,measpardim),save::slfsmstar
double precision,dimension(scarraydimension,measpardim),save::slfsfreq
double precision,dimension(scarraydimension,measpardim),save::slfsavgx
double precision,dimension(scarraydimension,measpardim),save::slfsavgy
double precision,dimension(scarraydimension,measpardim),save::slfsstdx
double precision,dimension(scarraydimension,measpardim),save::slfsstdy
double precision,dimension(scarraydimension,measpardim),save::slfsmaxx
double precision,dimension(scarraydimension,measpardim),save::slfsmaxy
double precision,dimension(scarraydimension,measpardim),save::slfsminx
double precision,dimension(scarraydimension,measpardim),save::slfsminy
logical,dimension(scarraydimension,measpardim),save::slfsnobif
double precision,save::minextfreq
double precision,save::freqsamefrac
double precision,save::avgsamefrac
integer(KIND=int8byte),save::numchunks
integer(KIND=int8byte),save::slice
integer(KIND=int8byte),save::fs
integer(KIND=int8byte),save::fs2
integer(KIND=int8byte),save::chunk
integer(KIND=int8byte),save::chunk2
integer(KIND=int8byte),save::cfs
double precision,dimension(measpardim2,scarraydimension),save::carea
integer(KIND=int8byte),dimension(measpardim2),save::numcfs
logical,dimension(measpardim2,scarraydimension),save::noprevarea
logical,dimension(measpardim2,scarraydimension),save::nonextarea
double precision,dimension(measpardim2,scarraydimension),save::cmstar
double precision,dimension(measpardim2,scarraydimension),save::cfreq
double precision,dimension(measpardim2,scarraydimension),save::corbtype
integer(KIND=int8byte),dimension(measpardim2,scarraydimension),save::corbnum
double precision,dimension(measpardim2,scarraydimension),save::cavgx
double precision,dimension(measpardim2,scarraydimension),save::cavgy
double precision,dimension(measpardim2,scarraydimension),save::cstdx
double precision,dimension(measpardim2,scarraydimension),save::cstdy
double precision,dimension(measpardim2,scarraydimension),save::cmaxx
double precision,dimension(measpardim2,scarraydimension),save::cmaxy
double precision,dimension(measpardim2,scarraydimension),save::cminx
double precision,dimension(measpardim2,scarraydimension),save::cminy
logical,dimension(measpardim2,scarraydimension),save::cnobif
integer(KIND=int8byte),dimension(measpardim2,scarraydimension),save::cfsfromslice
double precision,dimension(measpardim2,scarraydimension),save::badnessoffit
double precision,save::badness
double precision,save::oldbadness
integer(KIND=int8byte),save::matchchunknum
logical,save::matchchunkocc
logical,dimension(50),save::mcond
integer(KIND=int8byte),save::numuniqoverlaps
logical,save::isextremal

logical,save::wehaveafloater
double precision,dimension(4),save::floatarea
double precision,dimension(4),save::floatmstar
double precision,dimension(4),save::floatfreq
double precision,dimension(4),save::floatorbtype
integer(KIND=int8byte),dimension(4),save::floatorbnum
double precision,dimension(4),save::floatavgx
double precision,dimension(4),save::floatavgy
double precision,dimension(4),save::floatstdx
double precision,dimension(4),save::floatstdy
double precision,dimension(4),save::floatmaxx
double precision,dimension(4),save::floatmaxy
double precision,dimension(4),save::floatminx
double precision,dimension(4),save::floatminy
logical,dimension(4),save::floatnobif
integer(KIND=int8byte),dimension(4),save::floatfsfromslice

integer(KIND=int8byte),save::numint
integer(KIND=int8byte),save::minnumint
double precision,save::theta
double precision,save::phi
double precision,save::bigd
double precision,save::p
double precision,save::q
double precision,save::u
double precision,save::c
double precision,save::s
double precision,save::ai
double precision,save::aii
double precision,save::aiii
double precision,save::bi
double precision,save::bii
double precision,save::biii
double precision,save::ci
double precision,save::cii
double precision,save::ciii

double precision,save::cp1
double precision,save::cp2
double precision,save::cp3
double precision,save::cfintpointx
double precision,save::cfintpointy
double precision,save::cfintpointz
double precision,save::csubtractorx
double precision,save::csubtractory
double precision,save::csubtractorz

double precision,dimension(measpardim2),save::ndfreqarray
double precision,dimension(measpardim2),save::ndmassarray
double precision,dimension(measpardim2),save::ndcurvarray
double precision,dimension(measpardim2),save::ndorbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::ndorbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::ndfromslicearray
double precision,dimension(measpardim2),save::ndavgxrucarray
double precision,dimension(measpardim2),save::ndavgyrucarray
double precision,dimension(measpardim2),save::ndavgzrucarray

double precision,dimension(measpardim2),save::ndcnfreqarray
double precision,dimension(measpardim2),save::ndcnmassarray
double precision,dimension(measpardim2),save::ndcncurvarray
double precision,dimension(measpardim2),save::ndcnorbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::ndcnorbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::ndcnfromslicearray
double precision,dimension(measpardim2),save::ndcnavgxrucarray
double precision,dimension(measpardim2),save::ndcnavgyrucarray
double precision,dimension(measpardim2),save::ndcnavgzrucarray

double precision,dimension(measpardim2),save::cntempfreqarray
double precision,dimension(measpardim2),save::cntempmassarray
double precision,dimension(measpardim2),save::cntempcurvarray
double precision,dimension(measpardim2),save::cntemporbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::cntemporbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::cntempfromslicearray
double precision,dimension(measpardim2),save::cntempavgxrucarray
double precision,dimension(measpardim2),save::cntempavgyrucarray
double precision,dimension(measpardim2),save::cntempavgzrucarray

double precision,dimension(measpardim2),save::sortedfreqarray
double precision,dimension(measpardim2),save::sortedmassarray
double precision,dimension(measpardim2),save::sortedcurvarray
double precision,dimension(measpardim2),save::sortedorbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::sortedorbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::sortedfromslicearray
double precision,dimension(measpardim2),save::sortedavgxrucarray
double precision,dimension(measpardim2),save::sortedavgyrucarray
double precision,dimension(measpardim2),save::sortedavgzrucarray

double precision,dimension(measpardim2),save::tempfreqarray
double precision,dimension(measpardim2),save::tempmassarray
double precision,dimension(measpardim2),save::tempcurvarray
double precision,dimension(measpardim2),save::temporbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::temporbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::tempfromslicearray
double precision,dimension(measpardim2),save::tempavgxrucarray
double precision,dimension(measpardim2),save::tempavgyrucarray
double precision,dimension(measpardim2),save::tempavgzrucarray
double precision,dimension(measpardim2),save::tempstdfreqarray
double precision,dimension(measpardim2),save::tempstdmassarray
double precision,dimension(measpardim2),save::tempstdcurvarray
double precision,dimension(measpardim2),save::tempstdorbtypearray
double precision,dimension(measpardim2),save::tempstdavgxrucarray
double precision,dimension(measpardim2),save::tempstdavgyrucarray
double precision,dimension(measpardim2),save::tempstdavgzrucarray
integer(KIND=int8byte),dimension(measpardim2),save::tempnumorbsarray

double precision,dimension(measpardim2),save::averagedfreqarray
double precision,dimension(measpardim2),save::averagedmassarray
double precision,dimension(measpardim2),save::averagedcurvarray
double precision,dimension(measpardim2),save::averagedorbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::averagedorbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::averagedfromslicearray
double precision,dimension(measpardim2),save::averagedavgxrucarray
double precision,dimension(measpardim2),save::averagedavgyrucarray
double precision,dimension(measpardim2),save::averagedavgzrucarray
double precision,dimension(measpardim2),save::stdfreqarray
double precision,dimension(measpardim2),save::stdmassarray
double precision,dimension(measpardim2),save::stdcurvarray
double precision,dimension(measpardim2),save::stdorbtypearray
double precision,dimension(measpardim2),save::stdavgxrucarray
double precision,dimension(measpardim2),save::stdavgyrucarray
double precision,dimension(measpardim2),save::stdavgzrucarray
integer(KIND=int8byte),dimension(measpardim2),save::numorbsarray

double precision,dimension(measpardim2),save::av2freqarray
double precision,dimension(measpardim2),save::av2massarray
double precision,dimension(measpardim2),save::av2curvarray
double precision,dimension(measpardim2),save::av2orbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::av2orbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::av2fromslicearray
double precision,dimension(measpardim2),save::av2avgxrucarray
double precision,dimension(measpardim2),save::av2avgyrucarray
double precision,dimension(measpardim2),save::av2avgzrucarray
double precision,dimension(measpardim2),save::av2stdfreqarray
double precision,dimension(measpardim2),save::av2stdmassarray
double precision,dimension(measpardim2),save::av2stdcurvarray
double precision,dimension(measpardim2),save::av2stdorbtypearray
double precision,dimension(measpardim2),save::av2stdavgxrucarray
double precision,dimension(measpardim2),save::av2stdavgyrucarray
double precision,dimension(measpardim2),save::av2stdavgzrucarray
integer(KIND=int8byte),dimension(measpardim2),save::av2numorbsarray

double precision,dimension(measpardim2),save::sr2freqarray
double precision,dimension(measpardim2),save::sr2massarray
double precision,dimension(measpardim2),save::sr2curvarray
double precision,dimension(measpardim2),save::sr2orbtypearray
integer(KIND=int8byte),dimension(measpardim2),save::sr2orbnumarray
integer(KIND=int8byte),dimension(measpardim2),save::sr2fromslicearray
double precision,dimension(measpardim2),save::sr2avgxrucarray
double precision,dimension(measpardim2),save::sr2avgyrucarray
double precision,dimension(measpardim2),save::sr2avgzrucarray
double precision,dimension(measpardim2),save::sr2stdfreqarray
double precision,dimension(measpardim2),save::sr2stdmassarray
double precision,dimension(measpardim2),save::sr2stdcurvarray
double precision,dimension(measpardim2),save::sr2stdorbtypearray
double precision,dimension(measpardim2),save::sr2stdavgxrucarray
double precision,dimension(measpardim2),save::sr2stdavgyrucarray
double precision,dimension(measpardim2),save::sr2stdavgzrucarray
integer(KIND=int8byte),dimension(measpardim2),save::sr2numorbsarray

logical,save::closex
logical,save::closenegx
logical,save::closeposx
logical,save::closey
logical,save::closenegy
logical,save::closeposy
logical,save::closez
logical,save::closenegz
logical,save::closeposz

integer(KIND=int8byte),save::cnsearchcounter
integer(KIND=int8byte),save::numsamecentre
integer(KIND=int8byte),save::numdiffcentre
integer(KIND=int8byte),save::numleftinarray
integer(KIND=int8byte),save::numnd
integer(KIND=int8byte),save::newnumnd
integer(KIND=int8byte),save::numtemp
integer(KIND=int8byte),save::numsorted
integer(KIND=int8byte),save::numaveraged
integer(KIND=int8byte),save::numav2
double precision,save::ndcnminfreq
logical,save::foundamin
double precision,save::freqavgsum
double precision,save::massavgsum
double precision,save::curvavgsum
double precision,save::orbtypeavgsum
double precision,save::avgxrucavgsum
double precision,save::avgyrucavgsum
double precision,save::avgzrucavgsum
double precision,save::freqstdsum
double precision,save::massstdsum
double precision,save::curvstdsum
double precision,save::orbtypestdsum
double precision,save::avgxrucstdsum
double precision,save::avgyrucstdsum
double precision,save::avgzrucstdsum

double precision,save::unitcellvolume
double precision,save::onekpointvolume

integer(KIND=int8byte),save::numrots
integer(KIND=int8byte),save::rotstepper
double precision,save::thetastart
double precision,save::thetaend
double precision,save::phistart
double precision,save::phiend
character(len=100),save::hvdstart
character(len=100),save::hvdend
integer(KIND=int8byte),save::floatloopcount

character(len=8),save::pdate
character(len=10),save::ptime
character(len=10),save::ptimezone
integer(KIND=int8byte),dimension(8),save::ptimedatevalues
integer(KIND=int8byte),save::dostimehr
integer(KIND=int8byte),save::dostimemin
integer(KIND=int8byte),save::dostimesec
integer(KIND=int8byte),save::dostimems
integer(KIND=int8byte),save::rottimehr
integer(KIND=int8byte),save::rottimemin
integer(KIND=int8byte),save::rottimesec
integer(KIND=int8byte),save::rottimems
integer(KIND=int8byte),save::dettimehr
integer(KIND=int8byte),save::dettimemin
integer(KIND=int8byte),save::dettimesec
integer(KIND=int8byte),save::dettimems
integer(KIND=int8byte),save::matchtimehr
integer(KIND=int8byte),save::matchtimemin
integer(KIND=int8byte),save::matchtimesec
integer(KIND=int8byte),save::matchtimems
integer(KIND=int8byte),save::tottimehr
integer(KIND=int8byte),save::tottimemin
integer(KIND=int8byte),save::tottimesec
integer(KIND=int8byte),save::tottimems
integer(KIND=int8byte),save::timeleftms

integer(KIND=int4byte),save::ncmdargs
integer(KIND=int4byte),save::iarg
character(len=50),dimension(10),save::arg
character(len=1),save::parauto
character(len=1),save::calcdos
character(len=1),save::calcext
character(len=1),save::fbxsf
logical,save::foundfbxsf

double precision,save::ebandmin
double precision,save::ebandmax
double precision,save::bandwidth
integer(KIND=int8byte),save::numtetrahedra
double precision,save::tetrahedronvolume

integer(KIND=int8byte)::i
integer(KIND=int8byte)::j
integer(KIND=int8byte)::k
double precision::percentdone
double precision::tetradoscontrib
integer(KIND=int8byte)::numoccupiedstates
integer(KIND=int8byte)::numemptystates
double precision::slicedoscontrib
integer(KIND=int8byte)::sliceocc
integer(KIND=int8byte)::sliceempty

integer(KIND=int8byte)::whichslice
integer(KIND=int8byte)::whichorb
double precision::bandelectronarea
double precision::bandholearea
double precision::sliceeleca
double precision::sliceholea

double precision,dimension(slicedimension),save::fsoutlinexext
double precision,dimension(slicedimension),save::fsoutlineyext
double precision,save::outlineavgxext
double precision,save::outlineavgyext
integer(KIND=int8byte),save::noutlinepts
double precision,save::outlinefreq

! Shows title and reads parameters from file if desired
write(*,'('' '')')
write(*,'('' ----------------------------------------------------------------------------- '')')
write(*,'(''   Supercell K-space Extremal Area Finder (SKEAF) '',A11,'', '',A12)')buildnumber,releasedate
write(*,'(''   Copyright (C) 2012 Patrick Rourke (rourke@gmail.com) '')')
write(*,'(''                                                        '')')
write(*,'(''   P.M.C. Rourke and S.R. Julian, Comput. Phys. Commun. 183, 324 (2012) '')')
write(*,'('' ----------------------------------------------------------------------------- '')')
write(*,'('' '')')
write(*,'('' Please see the README-forSKEAF.txt file for important instructions and notes!'')')

parauto='n'
parsread='u'
parcorrect='n'
calcdos='y'
calcext='y'
fbxsf='n'

ncmdargs=IARGC()
if (ncmdargs>10) then
 write(*,'('' '')')
 write(*,'('' Too many command line arguments; some might have been ignored.'')')
 ncmdargs=10
end if

if (ncmdargs>0) then
 do iarg=1,ncmdargs
  call GETARG(iarg,arg(iarg))
  if (arg(iarg)=='-nodos') then
   calcdos='n'
  end if
  if (arg(iarg)=='-noext') then
   calcext='n'
  end if
  if (arg(iarg)=='-rdcfg') then
   parsread='y'
   parauto='y'
  end if
  if (arg(iarg)=='-fbxsf') then
   fbxsf='y'
  end if
 end do

 if (calcdos=='n') then
  write(*,'('' '')')
  write(*,'('' Skipping Density of States calculation! '')')
 end if

 if (calcext=='n') then
  write(*,'('' '')')
  write(*,'('' Skipping Extremal Area Finder algorithm! '')')
 end if

 if ((parsread=='y').AND.(parauto=='y')) then
  write(*,'('' '')')
  write(*,'('' Auto-read parameters from config.in! '')')
  if (fbxsf=='y') then
   write(*,'('' BXSF filename in config.in used; filename from command line ignored. '')')
   fbxsf='n'
  end if
 end if

 if (fbxsf=='y') then
  foundfbxsf=.FALSE.
  do iarg=1,(ncmdargs-1)
   if ((foundfbxsf.eqv..FALSE.).AND.(arg(iarg)=='-fbxsf').AND.(arg(iarg+1)/='-nodos').AND.(arg(iarg+1)/='-noext')) then
    filename=arg(iarg+1)
    foundfbxsf=.TRUE.
   end if
  end do
  if (foundfbxsf.eqv..FALSE.) then
   do iarg=1,ncmdargs
    if ((foundfbxsf.eqv..FALSE.).AND.(arg(iarg)/='-fbxsf').AND.(arg(iarg)/='-nodos').AND.(arg(iarg)/='-noext')) then
     filename=arg(iarg)
     foundfbxsf=.TRUE.
    end if
   end do
  end if
  if (foundfbxsf.eqv..FALSE.) then
   write(*,'('' '')')
   write(*,'('' Missing BXSF filename on command line. '')')
   fbxsf='n'
  end if
 end if

end if

do while (parcorrect=='n')
 do while (parsread/='y'.AND.parsread/='n')
  write(*,'('' '')')
  write(*,'('' Read parameters from config.in? [y or n] '')')
  read(*,*)parsread
 end do

 if (parsread=='y') then
  open(14,file='config.in',status='old')
  read(14,'(A50)')filename
  read(14,'(F12.6)')fermienergy
  read(14,'(I4)')numint
  read(14,'(F10.6)')theta
  read(14,'(F10.6)')phi
  read(14,'(A1)')hvd
  read(14,'(F10.6)')minextfreq
  read(14,'(F7.3)')freqsamefrac
  read(14,'(F7.3)')avgsamefrac
  read(14,'(A1)')allowextnearwalls
  read(14,'(F10.6)')thetastart
  read(14,'(F10.6)')thetaend
  read(14,'(F10.6)')phistart
  read(14,'(F10.6)')phiend
  read(14,'(I5)')numrots

  ! Convert angles from degrees to radians
  theta=pi*theta/180.0D0
  phi=pi*phi/180.0D0
  thetastart=pi*thetastart/180.0D0
  phistart=pi*phistart/180.0D0
  thetaend=pi*thetaend/180.0D0
  phiend=pi*phiend/180.0D0

  close(14,status='keep')

  if ((fbxsf=='y').AND.(parauto=='n')) then
   write(*,'('' BXSF filename in config.in used; filename from command line ignored. '')')
   fbxsf='n'
  end if

 end if

 ! This section reads in the BXSF FS file
 if ((parsread=='n').AND.(fbxsf=='n')) then
  write(*,'('' '')')
  write(*,'('' XCrysDen FS filename? [50 chars. max, type 0 for inputfs.bxsf] '')')
  read(*,*)filename
  if (filename=='0') then
   filename='inputfs.bxsf'
  end if
 end if

 write(*,'('' '')')
 write(*,'('' Reading BXSF file: '',A50)')filename

 CALL preadbxsf

 if (numread/=numkpoints) then
  write(*,'('' '')')
  write(*,'('' ***BXSF ERROR: Number of energies read from file differs from number in file header!***'')')
  write(*,'('' '')')
  STOP
 end if

 ! This section lets the user choose a Fermi energy
 if (parsread=='n') then
  write(*,'('' '')')
  write(*,'('' Fermi energy? [type 0 for '',f12.6 ,'' Ryd] '')')readfermienergy
  read(*,*)fermienergy
  if (fermienergy==0.0D0) then
   fermienergy=readfermienergy
  end if
  write(*,'('' '')')
  write(*,'('' Fermi energy: '',f12.6, '' Ryd '')')fermienergy
 end if

 badinput=.FALSE.
 runonce=.FALSE.

 ! This section will let the user apply additional interpolation to the k-grid
 do while ((badinput.eqv..TRUE.).OR.(runonce.eqv..FALSE.))
  runonce=.TRUE.
  badinput=.FALSE.
  if (parsread=='n') then
   numintkpoints=((4*(minnumint-1))+1)*((4*(minnumint-1))+1)*((4*(minnumint-1))+1)
   write(*,'('' '')')
   write(*,'('' In the original reciprocal unit cell, we have: '')')
   write(*,'('' nx = '',I4, '', ny = '',I4, '', nz = '',I4)')nx,ny,nz
   write(*,'('' Total number of k-points =   '', I11)')numkpoints
   write(*,'('' '')')
   write(*,'('' Number of interpolated k-points per single new cell side? ['',I4 ,''-'',I4 ,''] '')')minnumint,maxnumint
   write(*,'('' (More interpolation gives more accurate results.) '')')
   read(*,*)numint
   if (numint<minnumint) then
    numint=minnumint
   end if
   if (numint>maxnumint) then
    numint=maxnumint
   end if
  end if
  numintkpoints=((4*(numint-1))+1)*((4*(numint-1))+1)*((4*(numint-1))+1)
  if (parsread=='n') then
   write(*,'('' '')')
   write(*,'('' Number of k-points per single new cell side =       '', I4)')numint
   write(*,'('' Number of k-points in single new cell =       '', I11)')numintkpoints/16
   write(*,'('' Number of k-points in supercell =             '', I11)')numintkpoints
   write(*,'('' Is this correct? [y or n]'')')
   read(*,*)intcorrect
   if (intcorrect/='y') then
    badinput=.TRUE.
   end if
  end if
  intkpointspacing=maxlreciplat/(numint-1)
  intkpointspacingx=intkpointspacing
  intkpointspacingy=intkpointspacing
 end do

 badinput=.FALSE.
 runonce=.FALSE.

 write(*,'('' '')')
 write(*,'('' Number of k-points per single new cell side = '', I4,&
&''; Interpolated kpointspacing = '',f11.8)')numint,intkpointspacing

 numslices=(4*(numint-1))+1
 numx=(4*(numint-1))+1
 numy=(4*(numint-1))+1
 write(slicelinefmt,'(''(A'',I4,'')'')')(numx+2)
 xkseparation=intkpointspacingx
 ykseparation=intkpointspacingy
 xlength=4.0D0*maxlreciplat
 ylength=4.0D0*maxlreciplat

 ! This section lets the user pick the H-vector direction
 do while ((badinput.eqv..TRUE.).OR.(runonce.eqv..FALSE.))
  runonce=.TRUE.
  badinput=.FALSE.
  if (parsread=='n') then
   write(*,'('' '')')
   write(*,'('' H-vector direction? [a, b, c, n for non-princ., or r for auto-rot.] '')')
   read(*,*)hvd

   CALL psetangle(hvd,theta,phi)

   ! ************************ auto rot.
   if (hvd=='r') then 

    write(*,'('' '')')
    write(*,'('' Starting H-vector direction? [a, b, c, or n for non-princ.] '')')
    read(*,*)hvdstart 

    CALL psetangle(hvdstart,thetastart,phistart)

    if (badinput.eqv..FALSE.) then
     write(*,'('' '')')
     write(*,'('' Ending H-vector direction? [a, b, c, or n for non-princ.] '')')
     read(*,*)hvdend

     CALL psetangle(hvdend,thetaend,phiend)

    end if

    if (badinput.eqv..FALSE.) then
     write(*,'('' '')')
     write(*,'('' Number of auto angles, including start and end? [2-99999] '')')
     read(*,*)numrots
     if (numrots<2) then
      numrots=2
     end if
     if (numrots>99999) then
      numrots=99999
     end if

     theta=thetastart
     phi=phistart 

     write(*,'('' '')')
     write(*,'('' Using '',I5,'' angles from theta = '',f10.6, &
& '', phi = '',f10.6, '' degrees to theta = '',f10.6,'', phi = '',f10.6,'' degrees'')') &
&numrots,180.0D0*thetastart/pi,180.0D0*phistart/pi,180.0D0*thetaend/pi, &
&180.0D0*phiend/pi
    end if

   else

    if (badinput.eqv..FALSE.) then
     write(*,'('' '')')
     write(*,'('' Using theta = '',f10.6, '' degrees, phi = '',f10.6, '' degrees '')') &
&180.0D0*theta/pi,180.0D0*phi/pi
     thetastart=theta
     thetaend=theta
     phistart=phi
     phiend=phi
     numrots=1
    end if

   end if
  end if
 end do

 badinput=.FALSE.
 runonce=.FALSE.

 ! Calculate the minimum fs area to be considered
 ! (2 k-points in the supercell)
 minarea = 2.0D0*xkseparation*ykseparation

 ! Ask for minimum extremal fs freq (in kT) to be allowed
 if (parsread=='n') then
  write(*,'('' '')')
  write(*,'('' Minimum allowed extremal Fermi surface dHvA frequency (in kT)? '')')
  write(*,'('' ['',F8.4,'' kT is a good guideline (2 k-points on RUC);'', &
&'' type 0 to allow all extremal frequencies] '')')freq2kmax
  read(*,*)minextfreq
 end if

 ! Ask for averaging parameters
 if (parsread=='n') then
  write(*,'('' '')')
  write(*,'('' Please set the parameters used for determining which orbits are multiple'', &
&'' copies of one another and therefore should be averaged together.'')')
  write(*,'('' Note that both conditions (frequencies almost equal and average coordinates'', &
&'' almost equal) must be true for particular similar orbits to be averaged.'')')
  write(*,'(''  Maximum fractional difference between frequencies for orbit averaging? [0-1] '')')
  write(*,'(''  [type 0.01 for default = 1%; 0 to disable averaging; 1 to average orbits '', &
&''of all frequencies (usually a bad idea)] '')')
  read(*,*)freqsamefrac
  write(*,'(''  Maximum distance (fraction of RUC side length) between average coordinates '', &
&''for orbit averaging? [0-1] '')')
  write(*,'(''  [type 0.05 for default = 5% of RUC side lengths; 0 to disable averaging; '', &
&''1 to average orbits of all locations (usual choice when running in auto-rot. mode)]'')')
  write(*,'(''  [Note: averaging orbits of all locations will make final listed coordinates meaningless.]'')')
  read(*,*)avgsamefrac
  if (freqsamefrac<0.0D0) then
   freqsamefrac=0.0D0
  end if
  if (freqsamefrac>1.0D0) then
   freqsamefrac=1.0D0
  end if
  if (avgsamefrac<0.0D0) then
   avgsamefrac=0.0D0
  end if
  if (avgsamefrac>1.0D0) then
   avgsamefrac=1.0D0
  end if
 end if

 ! Ask if extremal orbits located near super-cell walls are allowed
 if (parsread=='n') then
  badinput=.FALSE.
  runonce=.FALSE.
  do while ((badinput.eqv..TRUE.).OR.(runonce.eqv..FALSE.))
   runonce=.TRUE.
   badinput=.FALSE.
   write(*,'('' '')')
   write(*,'('' Allow extremal orbits located near super-cell walls? [y or n] '')')
   write(*,'('' [type n for conservative default; y to allow these orbits, at risk'', &
&'' of introducing false frequencies based on orbit mis-matches]'')')
   read(*,*)allowextnearwalls
   if (allowextnearwalls/='y'.AND.allowextnearwalls/='n') then
    badinput=.TRUE.
   end if
  end do
 end if

 badinput=.FALSE.
 runonce=.FALSE.

 ! Ask if parameters are correct, then save config.in
 if (parauto=='y') then
  parcorrect='y'
 else
  do while ((badinput.eqv..TRUE.).OR.(runonce.eqv..FALSE.))
   runonce=.TRUE.
   badinput=.FALSE.
   write(*,'('' '')')
   write(*,'('' PARAMETER SUMMARY '')')
   write(*,'('' XCrysDen FS filename: '',A50)')filename
   write(*,'('' Fermi energy: '',f12.6, '' Ryd '')')fermienergy
   write(*,'('' Original     nx = '',I4,''  ny = '',I4,''  nz = '',I4,''   numkpoints = '',I11)')nx,ny,nz,numkpoints
   write(*,'('' New      numint = '',I4,''                       numkpoints = '',I11)')numint,numintkpoints
   if (hvd=='r') then
    write(*,'('' H-vector direction: '',A1,''   Number of auto rotated angles = '',I5)')hvd,numrots
    write(*,'('' Theta = '',F10.6,'' to '',F10.6,'' degrees;  Phi = '',F10.6,'' to '',F10.6,'' degrees '')') &
&thetastart*180.0D0/pi,thetaend*180.0D0/pi,phistart*180.0D0/pi,phiend*180.0D0/pi
   else
    write(*,'('' H-vector direction: '',A1,''   Theta = '',F10.6,'' degrees   Phi = '',F10.6,'' degrees '')') &
&hvd,theta*180.0D0/pi,phi*180.0D0/pi
   end if
   write(*,'('' Minimum extremal FS freq.: '',F8.4, '' kT '')')minextfreq
   write(*,'('' Maximum fractional diff. between orbit freqs. for averaging: '',F7.3)')freqsamefrac
   write(*,'('' Maximum distance (fraction of RUC side length) between orbit avg. coords. for averaging: '',F7.3)')&
&avgsamefrac
   if (allowextnearwalls=='y') then
    write(*,'('' Extremal orbits near super-cell walls are ALLOWED to be included in the output.'')')
   else
    write(*,'('' Extremal orbits near super-cell walls are REJECTED from the output.'')')
   end if
   write(*,'('' '')')
   write(*,'('' Is this correct? [y or n]'')')
   read(*,*)parcorrect
   if (parcorrect/='y'.AND.parcorrect/='n') then
    write(*,'('' Bad input '')')
    badinput=.TRUE.
   end if
   if (parcorrect=='n') then
    write(*,'('' '')')
    write(*,'('' Start again!'')')
    parsread='u'
   end if
  end do
 end if
end do

CALL DATE_AND_TIME(date=pdate,time=ptime,zone=ptimezone,values=ptimedatevalues)

if (parsread=='n') then
 open(16,file='config.in')
 rewind 16
 write(16,'(A50,2x,''| Filename (50 chars. max)'')')filename
 write(16,'(f12.6,40x,''| Fermi energy (Ryd)'')')fermienergy
 write(16,'(I4,48x,''| Interpolated number of points per single side'')')numint
 write(16,'(F10.6,42x,''| Theta (degrees)'')')180.0D0*theta/pi
 write(16,'(F10.6,42x,''| Phi (degrees)'')')180.0D0*phi/pi
 write(16,'(A1,51x,''| H-vector direction'')')hvd
 write(16,'(F8.4,44x,''| Minimum extremal FS freq. (kT)'')')minextfreq
 write(16,'(f7.3,45x,''| Maximum fractional diff. between orbit freqs. for averaging'')')freqsamefrac
 write(16,'(f7.3,45x,''| Maximum distance between orbit avg. coords. for averaging'')')avgsamefrac
 write(16,'(A1,51x,''| Allow extremal orbits near super-cell walls?'')')allowextnearwalls
 write(16,'(F10.6,42x,''| Starting theta (degrees)'')')180.0D0*thetastart/pi
 write(16,'(F10.6,42x,''| Ending theta (degrees)'')')180.0D0*thetaend/pi
 write(16,'(F10.6,42x,''| Starting phi (degrees)'')')180.0D0*phistart/pi
 write(16,'(F10.6,42x,''| Ending phi (degrees)'')')180.0D0*phiend/pi
 write(16,'(I5,47x,''| Number of rotation angles'')')numrots
 endfile 16
 close(16,status='keep')
end if

open(17,file='results_long.out')
rewind 17
write(17,'('' Long results file generated by S.K.E.A.F. '',A11)')buildnumber
write(17,'('' '')')
write(17,'('' XCrysDen FS filename: '',A50)')filename
write(17,'('' Fermi energy: '',f12.6, '' Ryd '')')fermienergy
write(17,'('' Original     nx = '',I4,''  ny = '',I4,''  nz = '',I4,''   numkpoints = '',I11)')nx,ny,nz,numkpoints
write(17,'('' New      numint = '',I4,''                       numkpoints = '',I11)')numint,numintkpoints
if (hvd=='r') then
 write(17,'('' H-vector direction: '',A1,''   Number of auto rotated angles = '',I5)')hvd,numrots
 write(17,'('' Theta = '',F10.6,'' to '',F10.6,'' degrees;  Phi = '',F10.6,'' to '',F10.6,'' degrees '')') &
&thetastart*180.0D0/pi,thetaend*180.0D0/pi,phistart*180.0D0/pi,phiend*180.0D0/pi
else
 write(17,'('' H-vector direction: '',A1,''   Theta = '',F10.6,'' degrees   Phi = '',F10.6,'' degrees '')') &
&hvd,theta*180.0D0/pi,phi*180.0D0/pi
end if
write(17,'('' Minimum extremal FS freq.: '',F8.4, '' kT '')')minextfreq
write(17,'('' Maximum fractional diff. between orbit freqs. for averaging: '',F7.3)')freqsamefrac
write(17,'('' Maximum distance (fraction of RUC side length) between orbit avg. coords. for averaging: '',F7.3)')&
&avgsamefrac
if (allowextnearwalls=='y') then
 write(17,'('' Extremal orbits near super-cell walls are ALLOWED to be included in the output.'')')
else
 write(17,'('' Extremal orbits near super-cell walls are REJECTED from the output.'')')
end if

open(18,file='results_short.out')
rewind 18
write(18,'('' Short results file generated by S.K.E.A.F. '',A11)')buildnumber
write(18,'('' '')')
write(18,'('' XCrysDen FS filename: '',A50)')filename
write(18,'('' Fermi energy: '',f12.6, '' Ryd '')')fermienergy
write(18,'('' Original     nx = '',I4,''  ny = '',I4,''  nz = '',I4,''   numkpoints = '',I11)')nx,ny,nz,numkpoints
write(18,'('' New      numint = '',I4,''                       numkpoints = '',I11)')numint,numintkpoints
if (hvd=='r') then
 write(18,'('' H-vector direction: '',A1,''   Number of auto rotated angles = '',I5)')hvd,numrots
 write(18,'('' Theta = '',F10.6,'' to '',F10.6,'' degrees;  Phi = '',F10.6,'' to '',F10.6,'' degrees '')') &
&thetastart*180.0D0/pi,thetaend*180.0D0/pi,phistart*180.0D0/pi,phiend*180.0D0/pi
else
 write(18,'('' H-vector direction: '',A1,''   Theta = '',F10.6,'' degrees   Phi = '',F10.6,'' degrees '')') &
&hvd,theta*180.0D0/pi,phi*180.0D0/pi
end if
write(18,'('' Minimum extremal FS freq.: '',F8.4, '' kT '')')minextfreq
write(18,'('' Maximum fractional diff. between orbit freqs. for averaging: '',F7.3)')freqsamefrac
write(18,'('' Maximum distance (fraction of RUC side length) between orbit avg. coords. for averaging: '',F7.3)')&
&avgsamefrac
if (allowextnearwalls=='y') then
 write(18,'('' Extremal orbits near super-cell walls are ALLOWED to be included in the output.'')')
else
 write(18,'('' Extremal orbits near super-cell walls are REJECTED from the output.'')')
end if

write(17,'('' '')')
write(17,'('' Original reciprocal lattice vector lengths (a, b, c) in a.u.^-1 (no 2Pi factor)'')')
write(17,'('' ('',f11.8,'', '',f11.8,'', '',f11.8,'')'')') &
&patricklreciplatx,patricklreciplaty,patricklreciplatz
write(17,'('' '')')
write(17,'('' Reciprocal lattice vector lengths (a, b, c) in Angstroms^-1 (2Pi factor included)'')')
write(17,'('' ('',f11.8,'', '',f11.8,'', '',f11.8,'')'')')lreciplatx,lreciplaty,lreciplatz
write(17,'('' '')')
write(17,'('' Default kpointspacings (a, b, c, min) in Angstroms^-1 (2Pi factor included)'')')
write(17,'('' ('',f11.8,'', '',f11.8,'', '',f11.8,'', '',f11.8,'')'')') &
&kpointspacingx,kpointspacingy,kpointspacingz,minkpointspacing
write(17,'('' '')')
write(17,'('' Actual used (interpolated) kpointspacing in Angstroms^-1 (2Pi factor included) = '',f11.8)')&
&intkpointspacing
write(17,'('' '')')
write(17,'('' Frequency for 2 k-points on original RUC faces (ab, ac, bc, max) in kT'')')
write(17,'(''('',f11.8,'', '',f11.8,'', '',f11.8,'', '',f11.8,'')'')')freq2kxy,freq2kxz,freq2kyz,freq2kmax

write(17,'('' '')')
write(17,'('' Started finding DOS on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'',I2,'':'',I2,''.'',I3.3)') &
&ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1),ptimedatevalues(5), &
&ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)

write(18,'('' '')')
write(18,'('' Started finding DOS on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'',I2,'':'',I2,''.'',I3.3)') &
&ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1),ptimedatevalues(5), &
&ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)

open(19,file='results_freqvsangle.out')
rewind 19
write(19,'(''Theta(deg),'',''Phi(deg),'',''Freq(kT),'',''mstar(me),'',''Curv(kTA2),'', &
&''Type(+e-h),'',''NumOrbCopy'')')

open(20,file='results_orbitoutlines_invAng.out')
rewind 20
if (hvd=='r') then
 write(20,'(''Note: In autorotation mode, detailed orbit outline information'', &
&'' is disabled. If you want this information, please do a single run at the angle of interest.'')')
end if

open(21,file='results_orbitoutlines_invau.out')
rewind 21
if (hvd=='r') then
 write(21,'(''Note: In autorotation mode, detailed orbit outline information'', &
&'' is disabled. If you want this information, please do a single run at the angle of interest.'')')
end if

!************************ calculate and print to screen bandwidth etc. ************

! find Ebandmin and Ebandmax (and therefore bandwidth)
ebandmin=masterkarray(1,1,1)
ebandmax=masterkarray(1,1,1)
do i=1,nx
 do j=1,ny
  do k=1,nz
   if (masterkarray(i,j,k)<ebandmin) then
   ebandmin=masterkarray(i,j,k)
   end if
   if (masterkarray(i,j,k)>ebandmax) then
   ebandmax=masterkarray(i,j,k)
   end if
  end do
 end do
end do
bandwidth=dabs(ebandmax-ebandmin)

! print Ebandmin, Ebandmax, and bandwidth to screen
write(17,'('' '')')
write(17,'('' Min band energy = '',f12.6,'' Ryd, max band energy = '',f12.6, &
&'' Ryd, bandwidth = '',f12.6,'' Ryd'')')ebandmin,ebandmax,bandwidth

if ((fermienergy<ebandmin).OR.(fermienergy>ebandmax)) then
 write(*,'('' '')')
 write(*,'('' This band does not cross the Fermi energy, therefore there'', &
&'' is no Fermi surface, no extremal orbits and DOS(E_F) = 0. '')')
 write(*,'('' Skipping remaining DOS and extremal orbit extraction routines. '')')
 write(17,'('' '')')
 write(17,'('' This band does not cross the Fermi energy, therefore there'', &
&'' is no Fermi surface, no extremal orbits and DOS(E_F) = 0. '')')
 write(17,'('' Skipping remaining DOS and extremal orbit extraction routines. '')')
 write(18,'('' '')')
 write(18,'('' This band does not cross the Fermi energy, therefore there'', &
&'' is no Fermi surface, no extremal orbits and DOS(E_F) = 0. '')')
 write(18,'('' Skipping remaining DOS and extremal orbit extraction routines. '')')
 calcdos='n'
 calcext='n'
end if

!************************************************************************************

!***XXX***XXX***XXX
! Here I find the density of states

CALL pgettime(dostimehr,dostimemin,dostimesec,dostimems)
CALL pgettime(tottimehr,tottimemin,tottimesec,tottimems)

if (calcdos=='n') then
 write(*,'('' '')')
 write(*,'('' Skipped DOS calculation. '')')
 write(17,'('' '')')
 write(17,'('' Skipped DOS calculation. '')')
 write(17,'('' '')')
 write(18,'('' '')')
 write(18,'('' Skipped DOS calculation. '')')
 write(18,'('' '')')
else

! *****************

unitcellvolume=dabs((plrx1*plry2*plrz3)-(plrx1*plry3*plrz2)- &
&(plrx2*plry1*plrz3)+(plrx2*plry3*plrz1)+(plrx3*plry1*plrz2)- &
&(plrx3*plry2*plrz1))
onekpointvolume=unitcellvolume/((dble(4*numint)-1.0D0)* &
&(dble(4*numint)-1.0D0)*(dble(4*numint)-1.0D0))

numoccupiedstates=0
numemptystates=0

! ************ Begin tetrahedron DOS calculation

numtetrahedra=6*((4*numint)-1)*((4*numint)-1)*((4*numint)-1)
tetrahedronvolume=unitcellvolume/dble(numtetrahedra)
tetradoscontrib=0.0D0

percentdone=0.0D0

write(*,'('' '')')
write(*,'('' Density of states calculation:   0.0 %''$)')

!$omp parallel do private(k,slicedoscontrib,sliceocc,sliceempty) &
!$omp shared(numint,percentdone) &
!$omp reduction(+:tetradoscontrib,numoccupiedstates,numemptystates)
do k=1,((4*numint)-1)

 percentdone=percentdone+((1.0D0/((4.0D0*dble(numint))-1.0D0))*100.0D0)
 write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,F5.1,'' %''$)')char(8),char(8),char(8),&
&char(8),char(8),char(8),char(8),percentdone

 CALL slicedos(k,slicedoscontrib,sliceocc,sliceempty)

 tetradoscontrib=tetradoscontrib+slicedoscontrib
 numoccupiedstates=numoccupiedstates+sliceocc
 numemptystates=numemptystates+sliceempty

end do
!$omp end parallel do

 write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,''100.0 %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8)

write(*,'('' '')')

! ************ End tetrahedron DOS calculation

 bandvolume=dble(numoccupiedstates)*onekpointvolume
 bandholevolume=dble(numemptystates)*onekpointvolume

! ****Estimate the number of electrons / holes in the band
! Note: this is just a simple estimate that assumes a fully occupied band has 2 electrons (ie. 1 up spin and 1 down spin),
! and 0 holes, while a fully unoccupied band has 0 electrons and 2 holes. In order to know which number (electrons or holes) to 
! pay attention to, one needs to know if the band is predominantly electron-like or hole-like

bandnumelectrons=2.0D0*bandvolume/unitcellvolume
bandnumholes=2.0D0*bandholevolume/unitcellvolume

write(*,'('' '')')
write(*,'('' Reciprocal Unit Cell volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points)   Vol of 1 k-point = '',F15.12,'' Angstroms^-3'')') &
&unitcellvolume,((4*numint)-1)*((4*numint)-1)*((4*numint)-1),onekpointvolume
write(*,'('' Band electron (occupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of electrons in this band** = '',F10.6)') &
&bandvolume,numoccupiedstates,bandnumelectrons
write(*,'('' Band hole (unoccupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of holes in this band** = '',F10.6)') &
&bandholevolume,numemptystates,bandnumholes
write(*,'('' (E_F - E_bandminimum)/E_bandwidth = '',F10.6)') &
&(fermienergy-ebandmin)/bandwidth
write(*,'('' **Either the estimated number of electrons (for an electron-like band)'', &
&'' or holes (for a hole-like band) calculated above should be used. '')')

write(17,'('' '')')
write(17,'('' Reciprocal Unit Cell volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points)   Vol of 1 k-point = '',F15.12,'' Angstroms^-3'')') &
&unitcellvolume,((4*numint)-1)*((4*numint)-1)*((4*numint)-1),onekpointvolume
write(17,'('' Band electron (occupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of electrons in this band** = '',F10.6)') &
&bandvolume,numoccupiedstates,bandnumelectrons
write(17,'('' Band hole (unoccupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of holes in this band** = '',F10.6)') &
&bandholevolume,numemptystates,bandnumholes
write(17,'('' (E_F - E_bandminimum)/E_bandwidth = '',F10.6)') &
&(fermienergy-ebandmin)/bandwidth
write(17,'('' **Either the estimated number of electrons (for an electron-like band)'', &
&'' or holes (for a hole-like band) calculated above should be used. '')')

write(18,'('' '')')
write(18,'('' Reciprocal Unit Cell volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points)   Vol of 1 k-point = '',F15.12,'' Angstroms^-3'')') &
&unitcellvolume,((4*numint)-1)*((4*numint)-1)*((4*numint)-1),onekpointvolume
write(18,'('' Band electron (occupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of electrons in this band** = '',F10.6)') &
&bandvolume,numoccupiedstates,bandnumelectrons
write(18,'('' Band hole (unoccupied) volume = '',F12.6,'' Angstroms^-3 ('',I11 &
&,'' k-points), Roughly estimated number of holes in this band** = '',F10.6)') &
&bandholevolume,numemptystates,bandnumholes
write(18,'('' (E_F - E_bandminimum)/E_bandwidth = '',F10.6)') &
&(fermienergy-ebandmin)/bandwidth
write(18,'('' **Either the estimated number of electrons (for an electron-like band)'', &
&'' or holes (for a hole-like band) calculated above should be used. '')')

!******************

write(*,'('' '')')
write(*,'('' Contribution of this band to the DOS(E_F) = '',F8.3 &
&,'' Angstroms^-3 Rydbergs^-1 per spin direction'')')tetradoscontrib
write(*,'('' (i.e. # of states per volume per energy per spin direction, at the Fermi energy)'')')

write(17,'('' '')')
write(17,'('' Contribution of this band to the DOS(E_F) = '',F8.3 &
&,'' Angstroms^-3 Rydbergs^-1 per spin direction'')')tetradoscontrib
write(17,'('' (i.e. # of states per volume per energy per spin direction, at the Fermi energy)'')')
write(17,'('' '')')

write(18,'('' '')')
write(18,'('' Contribution of this band to the DOS(E_F) = '',F8.3 &
&,'' Angstroms^-3 Rydbergs^-1 per spin direction'')')tetradoscontrib
write(18,'('' (i.e. # of states per volume per energy per spin direction, at the Fermi energy)'')')
write(18,'('' '')')

!***XXX***XXX***XXX

end if

CALL ptimediff(dostimehr,dostimemin,dostimesec,dostimems)

write(17,'('' DOS calculation took '',I2,''hr:'',I2,''min:'',I2,''.'',I3.3,''s'')') &
&dostimehr,dostimemin,dostimesec,dostimems
write(17,'('' '')')
write(18,'('' DOS calculation took '',I2,''hr:'',I2,''min:'',I2,''.'',I3.3,''s'')') &
&dostimehr,dostimemin,dostimesec,dostimems
write(18,'('' '')')

write(17,'('' Started main FS algorithm on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'',I2,'':'',I2,''.'',I3.3)') &
&ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1),ptimedatevalues(5), &
&ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)
write(18,'('' Started main FS algorithm on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'',I2,'':'',I2,''.'',I3.3)') &
&ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1),ptimedatevalues(5), &
&ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)

if (calcext=='n') then
 write(*,'('' '')')
 write(*,'('' Skipped extremal area finding algorithm. '')')
 write(*,'('' '')')
 write(17,'('' '')')
 write(17,'('' Skipped extremal area finding algorithm. '')')
 write(17,'('' '')')
 write(18,'('' '')')
 write(18,'('' Skipped extremal area finding algorithm. '')')
 write(18,'('' '')')
else

CALL pgettime(rottimehr,rottimemin,rottimesec,rottimems)

if (hvd=='r') then

 write(17,'('' '')')
 write(17,'('' Note: In autorotation mode, detailed long results information'', &
&'' is disabled. If you want this information, please do a single run at the angle of interest. '')')

end if

rotelectronarea=0.0D0
rotholearea=0.0D0

do rotstepper=1,numrots

 timeleftms=(((rottimehr*3600000)+(rottimemin*60000)+(rottimesec*1000)+rottimems)*(numrots + 1 - rotstepper))

 if (hvd=='r') then
  theta=thetastart+((dble(rotstepper)-1.0D0)*(thetaend-thetastart)/(dble(numrots)-1.0D0))
  phi=phistart+((dble(rotstepper)-1.0D0)*(phiend-phistart)/(dble(numrots)-1.0D0))

  write(17,'('' '')')
  write(17,'('' ------------------------------ '')')
  write(17,'('' ANGLE '',I5,'' of '',I5,''; theta = '',F10.6,'', phi = '',F10.6,'' degrees'')') &
&rotstepper,numrots,theta*180.0D0/pi,phi*180.0D0/pi

  write(18,'('' '')')
  write(18,'('' ------------------------------ '')')
  write(18,'('' ANGLE '',I5,'' of '',I5,''; theta = '',F10.6,'', phi = '',F10.6,'' degrees'')') &
&rotstepper,numrots,theta*180.0D0/pi,phi*180.0D0/pi

  write(*,'('' '')')
  write(*,'('' ------------------------------ '')')
  if (rotstepper==1) then
   write(*,'('' ANGLE '',I5,'' of '',I5,'';      unknown time left'')')rotstepper,numrots
  else
   write(*,'('' ANGLE '',I5,'' of '',I5,'';      minutes left ~ '',F7.3)') &
&rotstepper,numrots,dble(timeleftms)/60000.0D0
  end if
 end if

 ai=(plry2*plrz3)-(plry3*plrz2)
 aii=(plry1*plrz3)-(plry3*plrz1)
 aiii=(plry1*plrz2)-(plry2*plrz1)
 bi=(plrz2*plrx3)-(plrz3*plrx2)
 bii=(plrz1*plrx3)-(plrz3*plrx1)
 biii=(plrz1*plrx2)-(plrz2*plrx1)
 ci=(plrx2*plry3)-(plrx3*plry2)
 cii=(plrx1*plry3)-(plrx3*plry1)
 ciii=(plrx1*plry2)-(plrx2*plry1)
 bigd=(ai*plrx1)-(aii*plrx2)+(aiii*plrx3)

 p=dsin(theta)
 q=dcos(theta)
 u=(1.0D0-dcos(phi))
 s=dsin(phi)
 c=dcos(phi)

 if (rotstepper==1) then
  CALL pgettime(dettimehr,dettimemin,dettimesec,dettimems)
 end if

 ! Primary program loops

 bandelectronarea=0.0D0
 bandholearea=0.0D0
 percentdone=0.0D0

 if (hvd/='r') then
  write(*,'('' '')')
 end if

 write(*,'('' Interpolating and detecting orbits:   0.0 %''$)')

!$omp parallel do private(whichslice,sliceeleca,sliceholea) &
!$omp shared(numslices,numslfs,percentdone) &
!$omp reduction(+:bandelectronarea,bandholearea)
 do whichslice=1,numslices

  ! Important! The line below stops strange carry-over behaviour during auto-rot
  numslfs(whichslice)=0

  percentdone=percentdone+(1.0D0/dble(numslices))*100.0D0
  write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,F5.1,'' %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8),percentdone

  CALL sliceext(whichslice,INT(0,int8byte),sliceeleca,sliceholea)
  bandelectronarea=bandelectronarea+sliceeleca
  bandholearea=bandholearea+sliceholea

 end do
!$omp end parallel do

 write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,''100.0 %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8)

 if (hvd/='r') then

  write(*,'('' '')')
  write(*,'('' Writing long results file:   0.0 %''$)')
  percentdone=0.0D0

  slicescale='*'
  do j=1,numx
   if (j==1) then
    slicescale=slicescale(1:len_trim(slicescale))//'0'
   else if (j==nint(1.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'1'
   else if (j==nint(2.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'2'
   else if (j==nint(3.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'3'
   else if (j==nint(4.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'4'
   else if (j==nint(5.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'5'
   else if (j==nint(6.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'6'
   else if (j==nint(7.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'7'
   else if (j==nint(8.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'8'
   else if (j==nint(9.0D0*((dble(numx)+1.0D0)/10.0D0))) then
    slicescale=slicescale(1:len_trim(slicescale))//'9'
   else if (j==numx) then
    slicescale=slicescale(1:len_trim(slicescale))//'0'
   else
    slicescale=slicescale(1:len_trim(slicescale))//'*'
   end if
  end do
  slicescale=slicescale(1:len_trim(slicescale))//'*'

  do whichslice=1,numslices

   percentdone=percentdone+(1.0D0/dble(numslices))*100.0D0
   write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,F5.1,'' %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8),percentdone

   write(17,'('' '')')
   write(17,'('' ------------------------------ '')')
   write(17,'('' SLICE '',I4,'' of '',I4)')whichslice,numslices
   write(17,'('' '')')

   write(17,slicelinefmt)slicescale
   do j=1,numy
    write(17,slicelinefmt)slicepic(whichslice,j)
   end do
   write(17,slicelinefmt)slicescale

   if (numslfs(whichslice)>0) then
    do whichorb=1,numslfs(whichslice)
     if (slfsorbtype(whichslice,whichorb)==1.0D0) then
      write(17,'('' '')')
      write(17,'('' Orbit '',I5,'': electron'')')whichorb
     elseif (slfsorbtype(whichslice,whichorb)==-1.0D0) then
      write(17,'('' '')')
      write(17,'('' Orbit '',I5,'': hole'')')whichorb
     else
      write(17,'('' '')')
      write(17,'('' Orbit '',I5,'': unknown orbit type; orbtype variable = '',f7.3)')whichorb,&
&slfsorbtype(whichslice,whichorb)
     end if
     write(17,'('' Area = '',F8.4,'' A^(-2)     Freq. = '',F8.4,'' kT     m* = '',F8.4,'' m_e'')') &
&slfsarea(whichslice,whichorb),slfsfreq(whichslice,whichorb),slfsmstar(whichslice,whichorb)
     write(17,'('' Average (x,y) = ('',F6.3,'' +/- '',F6.3,'', '',F6.3,'' +/- '',F6.3 &
&,'') in SC fractional coordinates'')')slfsavgx(whichslice,whichorb),slfsstdx(whichslice,whichorb),&
&slfsavgy(whichslice,whichorb),slfsstdy(whichslice,whichorb)
    end do
   end if

   if ((numslfs(whichslice)==0).AND.(numsltoosmall(whichslice)==0).AND.(numslopen(whichslice)==0)) then
     write(17,'('' '')')
     write(17,'('' No orbits. '')')  
   else
    if (numsltoosmall(whichslice)>0) then
     write(17,'('' '')')
     write(17,'('' Number of ignored orbits smaller than 2 supercell points: '',I5)') &
&numsltoosmall(whichslice)
    end if
    if (numslopen(whichslice)>0) then
     write(17,'('' '')')
     write(17,'('' Number of ignored open orbits: '',I5)')numslopen(whichslice)
    end if
   end if

   write(17,'('' ------------------------------ '')')
  end do

  write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,''100.0 %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8)

 end if

 write(*,'('' '')')

 write(*,'('' Beginning Fermi surface matching... '')')

 if (rotstepper==1) then
  CALL ptimediff(dettimehr,dettimemin,dettimesec,dettimems)
  CALL pgettime(matchtimehr,matchtimemin,matchtimesec,matchtimems)
 end if

 !FS MATCHER

 numchunks=0
 wehaveafloater=.FALSE.

 do slice=1,numslices

  do fs=1,numslfs(slice)
   slfsnobif(slice,fs)=.TRUE.
  end do

  if (numchunks>0) then

! look for and mark forward bifurcations
   do chunk=1,numchunks
    if ((cfsfromslice(chunk,numcfs(chunk)))==(slice-1)) then
     numuniqoverlaps=0

     do fs=1,numslfs(slice)

      mcond(29)=(slfsminx(slice,fs)<cmaxx(chunk,numcfs(chunk)))
      mcond(30)=(slfsmaxx(slice,fs)>cminx(chunk,numcfs(chunk)))
      mcond(31)=(slfsminy(slice,fs)<cmaxy(chunk,numcfs(chunk)))
      mcond(32)=(slfsmaxy(slice,fs)>cminy(chunk,numcfs(chunk)))

      if (mcond(29).AND.mcond(30).AND.mcond(31).AND.mcond(32)) then

       oldbadness=((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))* &
&((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))
       oldbadness=oldbadness+(((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs)))* &
&((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs))))
       oldbadness=oldbadness+(((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs)))* &
&((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs))))
       oldbadness=oldbadness+(((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs)))* &
&((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs))))
       oldbadness=oldbadness+(((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs)))* &
&((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs))))
       oldbadness=oldbadness+(((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs)))* &
&((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs))))

       mcond(20)=.FALSE.

       do fs2=1,numslfs(slice)
        mcond(21)=(slfsminx(slice,fs2)<slfsmaxx(slice,fs))
        mcond(22)=(slfsmaxx(slice,fs2)>slfsminx(slice,fs))
        mcond(23)=(slfsminy(slice,fs2)<slfsmaxy(slice,fs))
        mcond(24)=(slfsmaxy(slice,fs2)>slfsminy(slice,fs))
        if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.(fs2/=fs)) then
         mcond(20)=.TRUE.
        end if
       end do

       do chunk2=1,numchunks
        if ((cfsfromslice(chunk2,numcfs(chunk2)))==(slice-1)) then
         mcond(21)=(slfsminx(slice,fs)<cmaxx(chunk2,numcfs(chunk2)))
         mcond(22)=(slfsmaxx(slice,fs)>cminx(chunk2,numcfs(chunk2)))
         mcond(23)=(slfsminy(slice,fs)<cmaxy(chunk2,numcfs(chunk2)))
         mcond(24)=(slfsmaxy(slice,fs)>cminy(chunk2,numcfs(chunk2)))
         if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.(chunk2/=chunk)) then
          mcond(20)=.TRUE.
         end if
        end if
       end do

       if (mcond(20).eqv..FALSE.) then
        numuniqoverlaps=numuniqoverlaps+1
        do fs2=1,numslfs(slice)
         mcond(21)=(slfsminx(slice,fs2)<cmaxx(chunk,numcfs(chunk)))
         mcond(22)=(slfsmaxx(slice,fs2)>cminx(chunk,numcfs(chunk)))
         mcond(23)=(slfsminy(slice,fs2)<cmaxy(chunk,numcfs(chunk)))
         mcond(24)=(slfsmaxy(slice,fs2)>cminy(chunk,numcfs(chunk)))
         mcond(25)=(slfsavgx(slice,fs2)<cmaxx(chunk,numcfs(chunk))) &
&.AND.(slfsavgx(slice,fs2)>cminx(chunk,numcfs(chunk)))
         mcond(26)=(slfsavgy(slice,fs2)<cmaxy(chunk,numcfs(chunk))) &
&.AND.(slfsavgy(slice,fs2)>cminy(chunk,numcfs(chunk)))
         mcond(27)=(cavgx(chunk,numcfs(chunk))<slfsmaxx(slice,fs2)) &
&.AND.(cavgx(chunk,numcfs(chunk))>slfsminx(slice,fs2))
         mcond(28)=(cavgy(chunk,numcfs(chunk))<slfsmaxy(slice,fs2)) &
&.AND.(cavgy(chunk,numcfs(chunk))>slfsminy(slice,fs2))
         badness=((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs2)))* &
&((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs2)))
         badness=badness+(((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs2)))* &
&((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs2))))
         badness=badness+(((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs2)))* &
&((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs2))))
         badness=badness+(((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs2)))* &
&((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs2))))
         badness=badness+(((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs2)))* &
&((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs2))))
         badness=badness+(((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs2)))* &
&((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs2))))
         if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.mcond(25) &
&.AND.mcond(26).AND.mcond(27).AND.mcond(28).AND.(fs2/=fs).AND.(badness<oldbadness)) then
          numuniqoverlaps=numuniqoverlaps+1
         end if
        end do
       end if

      end if

     end do

     if (numuniqoverlaps>1) then
      cnobif(chunk,numcfs(chunk))=.FALSE.
     end if

    end if
   end do

! look for and mark reverse bifurcations
   do fs=1,numslfs(slice)
    numuniqoverlaps=0

    do chunk=1,numchunks
     if ((cfsfromslice(chunk,numcfs(chunk)))==(slice-1)) then
  
      mcond(29)=(slfsminx(slice,fs)<cmaxx(chunk,numcfs(chunk)))
      mcond(30)=(slfsmaxx(slice,fs)>cminx(chunk,numcfs(chunk)))
      mcond(31)=(slfsminy(slice,fs)<cmaxy(chunk,numcfs(chunk)))
      mcond(32)=(slfsmaxy(slice,fs)>cminy(chunk,numcfs(chunk)))

      if (mcond(29).AND.mcond(30).AND.mcond(31).AND.mcond(32)) then

       oldbadness=((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))* &
&((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))
       oldbadness=oldbadness+(((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs)))* &
&((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs))))
       oldbadness=oldbadness+(((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs)))* &
&((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs))))
       oldbadness=oldbadness+(((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs)))* &
&((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs))))
       oldbadness=oldbadness+(((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs)))* &
&((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs))))
       oldbadness=oldbadness+(((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs)))* &
&((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs))))

       mcond(20)=.FALSE.

       do chunk2=1,numchunks
        if ((cfsfromslice(chunk2,numcfs(chunk2)))==(slice-1)) then
         mcond(21)=(cminx(chunk2,numcfs(chunk2))<cmaxx(chunk,numcfs(chunk)))
         mcond(22)=(cmaxx(chunk2,numcfs(chunk2))>cminx(chunk,numcfs(chunk)))
         mcond(23)=(cminy(chunk2,numcfs(chunk2))<cmaxy(chunk,numcfs(chunk)))
         mcond(24)=(cmaxy(chunk2,numcfs(chunk2))>cminy(chunk,numcfs(chunk)))
         if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.(chunk2/=chunk)) then
          mcond(20)=.TRUE.
         end if
        end if
       end do

       do fs2=1,numslfs(slice)
        mcond(21)=(slfsminx(slice,fs2)<cmaxx(chunk,numcfs(chunk)))
        mcond(22)=(slfsmaxx(slice,fs2)>cminx(chunk,numcfs(chunk)))
        mcond(23)=(slfsminy(slice,fs2)<cmaxy(chunk,numcfs(chunk)))
        mcond(24)=(slfsmaxy(slice,fs2)>cminy(chunk,numcfs(chunk)))
        if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.(fs2/=fs)) then
         mcond(20)=.TRUE.
        end if
       end do

       if (mcond(20).eqv..FALSE.) then
        numuniqoverlaps=numuniqoverlaps+1
        do chunk2=1,numchunks
         if ((cfsfromslice(chunk2,numcfs(chunk2)))==(slice-1)) then
          mcond(21)=(slfsminx(slice,fs)<cmaxx(chunk2,numcfs(chunk2)))
          mcond(22)=(slfsmaxx(slice,fs)>cminx(chunk2,numcfs(chunk2)))
          mcond(23)=(slfsminy(slice,fs)<cmaxy(chunk2,numcfs(chunk2)))
          mcond(24)=(slfsmaxy(slice,fs)>cminy(chunk2,numcfs(chunk2)))
          mcond(25)=(slfsavgx(slice,fs)<cmaxx(chunk2,numcfs(chunk2))) &
&.AND.(slfsavgx(slice,fs)>cminx(chunk2,numcfs(chunk2)))
          mcond(26)=(slfsavgy(slice,fs)<cmaxy(chunk2,numcfs(chunk2))) &
&.AND.(slfsavgy(slice,fs)>cminy(chunk2,numcfs(chunk2)))
          mcond(27)=(cavgx(chunk2,numcfs(chunk2))<slfsmaxx(slice,fs)) &
&.AND.(cavgx(chunk2,numcfs(chunk2))>slfsminx(slice,fs))
          mcond(28)=(cavgy(chunk2,numcfs(chunk2))<slfsmaxy(slice,fs)) &
&.AND.(cavgy(chunk2,numcfs(chunk2))>slfsminy(slice,fs))
          badness=((cavgx(chunk2,numcfs(chunk2)))-(slfsavgx(slice,fs)))* &
&((cavgx(chunk2,numcfs(chunk2)))-(slfsavgx(slice,fs)))
          badness=badness+(((cavgy(chunk2,numcfs(chunk2)))-(slfsavgy(slice,fs)))* &
&((cavgy(chunk2,numcfs(chunk2)))-(slfsavgy(slice,fs))))
          badness=badness+(((cmaxx(chunk2,numcfs(chunk2)))-(slfsmaxx(slice,fs)))* &
&((cmaxx(chunk2,numcfs(chunk2)))-(slfsmaxx(slice,fs))))
          badness=badness+(((cmaxy(chunk2,numcfs(chunk2)))-(slfsmaxy(slice,fs)))* &
&((cmaxy(chunk2,numcfs(chunk2)))-(slfsmaxy(slice,fs))))
          badness=badness+(((cminx(chunk2,numcfs(chunk2)))-(slfsminx(slice,fs)))* &
&((cminx(chunk2,numcfs(chunk2)))-(slfsminx(slice,fs))))
          badness=badness+(((cminy(chunk2,numcfs(chunk2)))-(slfsminy(slice,fs)))* &
&((cminy(chunk2,numcfs(chunk2)))-(slfsminy(slice,fs))))
          if (mcond(21).AND.mcond(22).AND.mcond(23).AND.mcond(24).AND.mcond(25) &
&.AND.mcond(26).AND.mcond(27).AND.mcond(28).AND.(chunk2/=chunk).AND.(badness<oldbadness)) then
           numuniqoverlaps=numuniqoverlaps+1
          end if
         end if
        end do
       end if

      end if

     end if

    end do

    if (numuniqoverlaps>1) then
     slfsnobif(slice,fs)=.FALSE.
    end if

   end do

  end if

! perform the actual matching, disallowing matching across detected bifurcations
  do fs=1,numslfs(slice)
 
   if (numchunks==0) then
    numchunks=1
    numcfs(1)=1
    noprevarea(1,1)=.TRUE.
    nonextarea(1,1)=.TRUE.
    carea(1,1)=slfsarea(slice,fs)
    cmstar(1,1)=slfsmstar(slice,fs)
    cfreq(1,1)=slfsfreq(slice,fs)
    corbtype(1,1)=slfsorbtype(slice,fs)
    corbnum(1,1)=slfsorbnum(slice,fs)
    cavgx(1,1)=slfsavgx(slice,fs)
    cavgy(1,1)=slfsavgy(slice,fs)
    cstdx(1,1)=slfsstdx(slice,fs)
    cstdy(1,1)=slfsstdy(slice,fs)
    cmaxx(1,1)=slfsmaxx(slice,fs)
    cmaxy(1,1)=slfsmaxy(slice,fs)
    cminx(1,1)=slfsminx(slice,fs)
    cminy(1,1)=slfsminy(slice,fs)
    cnobif(1,1)=.TRUE.
    cfsfromslice(1,1)=slice
    badnessoffit(1,1)=0.0D0

   else

    matchchunknum=0
    matchchunkocc=.FALSE.
    oldbadness=1000000.0D0
    do chunk=1,numchunks

     if ((cfsfromslice(chunk,numcfs(chunk))==slice).AND.(numcfs(chunk)>1)) then
      mcond(1)=(slfsavgx(slice,fs)<cavgx(chunk,numcfs(chunk)-1)+1.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(slfsavgx(slice,fs)>cavgx(chunk,numcfs(chunk)-1)-1.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(2)=(slfsavgy(slice,fs)<cavgy(chunk,numcfs(chunk)-1)+1.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(slfsavgy(slice,fs)>cavgy(chunk,numcfs(chunk)-1)-1.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(3)=(slfsmaxx(slice,fs)<cmaxx(chunk,numcfs(chunk)-1)+2.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(slfsmaxx(slice,fs)>cmaxx(chunk,numcfs(chunk)-1)-2.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(4)=(slfsmaxy(slice,fs)<cmaxy(chunk,numcfs(chunk)-1)+2.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(slfsmaxy(slice,fs)>cmaxy(chunk,numcfs(chunk)-1)-2.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(5)=(slfsminx(slice,fs)<cminx(chunk,numcfs(chunk)-1)+2.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(slfsminx(slice,fs)>cminx(chunk,numcfs(chunk)-1)-2.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(6)=(slfsminy(slice,fs)<cminy(chunk,numcfs(chunk)-1)+2.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(slfsminy(slice,fs)>cminy(chunk,numcfs(chunk)-1)-2.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(7)=(slfsorbtype(slice,fs)==corbtype(chunk,numcfs(chunk)-1))
      mcond(8)=(cavgx(chunk,numcfs(chunk)-1)<slfsavgx(slice,fs)+1.0D0*slfsstdx(slice,fs)) &
&.AND.(cavgx(chunk,numcfs(chunk)-1)>slfsavgx(slice,fs)-1.0D0*slfsstdx(slice,fs))
      mcond(9)=(cavgy(chunk,numcfs(chunk)-1)<slfsavgy(slice,fs)+1.0D0*slfsstdy(slice,fs)) &
&.AND.(cavgy(chunk,numcfs(chunk)-1)>slfsavgy(slice,fs)-1.0D0*slfsstdy(slice,fs))
      mcond(10)=(cmaxx(chunk,numcfs(chunk)-1)<slfsmaxx(slice,fs)+2.0D0*slfsstdx(slice,fs)) &
&.AND.(cmaxx(chunk,numcfs(chunk)-1)>slfsmaxx(slice,fs)-2.0D0*slfsstdx(slice,fs))
      mcond(11)=(cmaxy(chunk,numcfs(chunk)-1)<slfsmaxy(slice,fs)+2.0D0*slfsstdy(slice,fs)) &
&.AND.(cmaxy(chunk,numcfs(chunk)-1)>slfsmaxy(slice,fs)-2.0D0*slfsstdy(slice,fs))
      mcond(12)=(cminx(chunk,numcfs(chunk)-1)<slfsminx(slice,fs)+2.0D0*slfsstdx(slice,fs)) &
&.AND.(cminx(chunk,numcfs(chunk)-1)>slfsminx(slice,fs)-2.0D0*slfsstdx(slice,fs))
      mcond(13)=(cminy(chunk,numcfs(chunk)-1)<slfsminy(slice,fs)+2.0D0*slfsstdy(slice,fs)) &
&.AND.(cminy(chunk,numcfs(chunk)-1)>slfsminy(slice,fs)-2.0D0*slfsstdy(slice,fs))
      mcond(14)=(slfsavgx(slice,fs)<cmaxx(chunk,numcfs(chunk)-1)) &
&.AND.(slfsavgx(slice,fs)>cminx(chunk,numcfs(chunk)-1))
      mcond(15)=(slfsavgy(slice,fs)<cmaxy(chunk,numcfs(chunk)-1)) &
&.AND.(slfsavgy(slice,fs)>cminy(chunk,numcfs(chunk)-1))
      mcond(16)=(cavgx(chunk,numcfs(chunk)-1)<slfsmaxx(slice,fs)) &
&.AND.(cavgx(chunk,numcfs(chunk)-1)>slfsminx(slice,fs))
      mcond(17)=(cavgy(chunk,numcfs(chunk)-1)<slfsmaxy(slice,fs)) &
&.AND.(cavgy(chunk,numcfs(chunk)-1)>slfsminy(slice,fs))
      mcond(18)=slfsnobif(slice,fs)
      mcond(19)=cnobif(chunk,numcfs(chunk)-1)
      if (mcond(1).AND.mcond(2).AND.mcond(3).AND.mcond(4).AND.mcond(5).AND.mcond(6).AND.mcond(7) &
&.AND.mcond(8).AND.mcond(9).AND.mcond(10).AND.mcond(11).AND.mcond(12).AND.mcond(13) &
&.AND.mcond(14).AND.mcond(15).AND.mcond(16).AND.mcond(17).AND.mcond(18).AND.mcond(19)) then
       badness=((cavgx(chunk,numcfs(chunk)-1))-(slfsavgx(slice,fs)))* &
&((cavgx(chunk,numcfs(chunk)-1))-(slfsavgx(slice,fs)))
       badness=badness+(((cavgy(chunk,numcfs(chunk)-1))-(slfsavgy(slice,fs)))* &
&((cavgy(chunk,numcfs(chunk)-1))-(slfsavgy(slice,fs))))
       badness=badness+(((cmaxx(chunk,numcfs(chunk)-1))-(slfsmaxx(slice,fs)))* &
&((cmaxx(chunk,numcfs(chunk)-1))-(slfsmaxx(slice,fs))))
       badness=badness+(((cmaxy(chunk,numcfs(chunk)-1))-(slfsmaxy(slice,fs)))* &
&((cmaxy(chunk,numcfs(chunk)-1))-(slfsmaxy(slice,fs))))
       badness=badness+(((cminx(chunk,numcfs(chunk)-1))-(slfsminx(slice,fs)))* &
&((cminx(chunk,numcfs(chunk)-1))-(slfsminx(slice,fs))))
       badness=badness+(((cminy(chunk,numcfs(chunk)-1))-(slfsminy(slice,fs)))* &
&((cminy(chunk,numcfs(chunk)-1))-(slfsminy(slice,fs))))
       if ((badness<badnessoffit(chunk,numcfs(chunk))).AND.(badness<oldbadness)) then
        oldbadness=badness
        matchchunkocc=.TRUE.
        matchchunknum=chunk
       end if
      end if
     else
      if ((cfsfromslice(chunk,numcfs(chunk)))==(slice-1)) then
       mcond(1)=(slfsavgx(slice,fs)<cavgx(chunk,numcfs(chunk))+1.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(slfsavgx(slice,fs)>cavgx(chunk,numcfs(chunk))-1.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(2)=(slfsavgy(slice,fs)<cavgy(chunk,numcfs(chunk))+1.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(slfsavgy(slice,fs)>cavgy(chunk,numcfs(chunk))-1.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(3)=(slfsmaxx(slice,fs)<cmaxx(chunk,numcfs(chunk))+2.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(slfsmaxx(slice,fs)>cmaxx(chunk,numcfs(chunk))-2.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(4)=(slfsmaxy(slice,fs)<cmaxy(chunk,numcfs(chunk))+2.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(slfsmaxy(slice,fs)>cmaxy(chunk,numcfs(chunk))-2.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(5)=(slfsminx(slice,fs)<cminx(chunk,numcfs(chunk))+2.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(slfsminx(slice,fs)>cminx(chunk,numcfs(chunk))-2.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(6)=(slfsminy(slice,fs)<cminy(chunk,numcfs(chunk))+2.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(slfsminy(slice,fs)>cminy(chunk,numcfs(chunk))-2.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(7)=(slfsorbtype(slice,fs)==corbtype(chunk,numcfs(chunk)))
       mcond(8)=(cavgx(chunk,numcfs(chunk))<slfsavgx(slice,fs)+1.0D0*slfsstdx(slice,fs)) &
&.AND.(cavgx(chunk,numcfs(chunk))>slfsavgx(slice,fs)-1.0D0*slfsstdx(slice,fs))
       mcond(9)=(cavgy(chunk,numcfs(chunk))<slfsavgy(slice,fs)+1.0D0*slfsstdy(slice,fs)) &
&.AND.(cavgy(chunk,numcfs(chunk))>slfsavgy(slice,fs)-1.0D0*slfsstdy(slice,fs))
       mcond(10)=(cmaxx(chunk,numcfs(chunk))<slfsmaxx(slice,fs)+2.0D0*slfsstdx(slice,fs)) &
&.AND.(cmaxx(chunk,numcfs(chunk))>slfsmaxx(slice,fs)-2.0D0*slfsstdx(slice,fs))
       mcond(11)=(cmaxy(chunk,numcfs(chunk))<slfsmaxy(slice,fs)+2.0D0*slfsstdy(slice,fs)) &
&.AND.(cmaxy(chunk,numcfs(chunk))>slfsmaxy(slice,fs)-2.0D0*slfsstdy(slice,fs))
       mcond(12)=(cminx(chunk,numcfs(chunk))<slfsminx(slice,fs)+2.0D0*slfsstdx(slice,fs)) &
&.AND.(cminx(chunk,numcfs(chunk))>slfsminx(slice,fs)-2.0D0*slfsstdx(slice,fs))
       mcond(13)=(cminy(chunk,numcfs(chunk))<slfsminy(slice,fs)+2.0D0*slfsstdy(slice,fs)) &
&.AND.(cminy(chunk,numcfs(chunk))>slfsminy(slice,fs)-2.0D0*slfsstdy(slice,fs))
       mcond(14)=(slfsavgx(slice,fs)<cmaxx(chunk,numcfs(chunk))) &
&.AND.(slfsavgx(slice,fs)>cminx(chunk,numcfs(chunk)))
       mcond(15)=(slfsavgy(slice,fs)<cmaxy(chunk,numcfs(chunk))) &
&.AND.(slfsavgy(slice,fs)>cminy(chunk,numcfs(chunk)))
       mcond(16)=(cavgx(chunk,numcfs(chunk))<slfsmaxx(slice,fs)) &
&.AND.(cavgx(chunk,numcfs(chunk))>slfsminx(slice,fs))
       mcond(17)=(cavgy(chunk,numcfs(chunk))<slfsmaxy(slice,fs)) &
&.AND.(cavgy(chunk,numcfs(chunk))>slfsminy(slice,fs))
       mcond(18)=slfsnobif(slice,fs)
       mcond(19)=cnobif(chunk,numcfs(chunk))
       if (mcond(1).AND.mcond(2).AND.mcond(3).AND.mcond(4).AND.mcond(5).AND.mcond(6).AND.mcond(7) &
&.AND.mcond(8).AND.mcond(9).AND.mcond(10).AND.mcond(11).AND.mcond(12).AND.mcond(13) &
&.AND.mcond(14).AND.mcond(15).AND.mcond(16).AND.mcond(17).AND.mcond(18).AND.mcond(19)) then
        badness=((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))* &
&((cavgx(chunk,numcfs(chunk)))-(slfsavgx(slice,fs)))
        badness=badness+(((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs)))* &
&((cavgy(chunk,numcfs(chunk)))-(slfsavgy(slice,fs))))
        badness=badness+(((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs)))* &
&((cmaxx(chunk,numcfs(chunk)))-(slfsmaxx(slice,fs))))
        badness=badness+(((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs)))* &
&((cmaxy(chunk,numcfs(chunk)))-(slfsmaxy(slice,fs))))
        badness=badness+(((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs)))* &
&((cminx(chunk,numcfs(chunk)))-(slfsminx(slice,fs))))
        badness=badness+(((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs)))* &
&((cminy(chunk,numcfs(chunk)))-(slfsminy(slice,fs))))
        if (badness<oldbadness) then
         oldbadness=badness
         matchchunkocc=.FALSE.
         matchchunknum=chunk
        end if
       end if
      end if
     end if
    end do

    if (matchchunknum==0) then
     numchunks=numchunks+1
     numcfs(numchunks)=1
     noprevarea(numchunks,1)=.TRUE.
     nonextarea(numchunks,1)=.TRUE.
     carea(numchunks,1)=slfsarea(slice,fs)
     cmstar(numchunks,1)=slfsmstar(slice,fs)
     cfreq(numchunks,1)=slfsfreq(slice,fs)
     corbtype(numchunks,1)=slfsorbtype(slice,fs)
     corbnum(numchunks,1)=slfsorbnum(slice,fs)
     cavgx(numchunks,1)=slfsavgx(slice,fs)
     cavgy(numchunks,1)=slfsavgy(slice,fs)
     cstdx(numchunks,1)=slfsstdx(slice,fs)
     cstdy(numchunks,1)=slfsstdy(slice,fs)
     cmaxx(numchunks,1)=slfsmaxx(slice,fs)
     cmaxy(numchunks,1)=slfsmaxy(slice,fs)
     cminx(numchunks,1)=slfsminx(slice,fs)
     cminy(numchunks,1)=slfsminy(slice,fs)
     cnobif(numchunks,1)=.TRUE.
     cfsfromslice(numchunks,1)=slice
     badnessoffit(numchunks,1)=0.0D0

    else
     if (matchchunkocc.eqv..FALSE.) then
      numcfs(matchchunknum)=numcfs(matchchunknum)+1
      carea(matchchunknum,numcfs(matchchunknum))=slfsarea(slice,fs)
      cmstar(matchchunknum,numcfs(matchchunknum))=slfsmstar(slice,fs)
      cfreq(matchchunknum,numcfs(matchchunknum))=slfsfreq(slice,fs)
      corbtype(matchchunknum,numcfs(matchchunknum))=slfsorbtype(slice,fs)
      corbnum(matchchunknum,numcfs(matchchunknum))=slfsorbnum(slice,fs)
      cavgx(matchchunknum,numcfs(matchchunknum))=slfsavgx(slice,fs)
      cavgy(matchchunknum,numcfs(matchchunknum))=slfsavgy(slice,fs)
      cstdx(matchchunknum,numcfs(matchchunknum))=slfsstdx(slice,fs)
      cstdy(matchchunknum,numcfs(matchchunknum))=slfsstdy(slice,fs)
      cmaxx(matchchunknum,numcfs(matchchunknum))=slfsmaxx(slice,fs)
      cmaxy(matchchunknum,numcfs(matchchunknum))=slfsmaxy(slice,fs)
      cminx(matchchunknum,numcfs(matchchunknum))=slfsminx(slice,fs)
      cminy(matchchunknum,numcfs(matchchunknum))=slfsminy(slice,fs)
      cnobif(matchchunknum,numcfs(matchchunknum))=.TRUE.
      cfsfromslice(matchchunknum,numcfs(matchchunknum))=slice
      badnessoffit(matchchunknum,numcfs(matchchunknum))=oldbadness
      nonextarea(matchchunknum,numcfs(matchchunknum))=.TRUE.
      noprevarea(matchchunknum,numcfs(matchchunknum))=.FALSE.
      nonextarea(matchchunknum,numcfs(matchchunknum)-1)=.FALSE.

     else

 !*************** put special floater section here (start)
      floatarea(1)=carea(matchchunknum,numcfs(matchchunknum))
      floatmstar(1)=cmstar(matchchunknum,numcfs(matchchunknum))
      floatfreq(1)=cfreq(matchchunknum,numcfs(matchchunknum))
      floatorbtype(1)=corbtype(matchchunknum,numcfs(matchchunknum))
      floatorbnum(1)=corbnum(matchchunknum,numcfs(matchchunknum))
      floatavgx(1)=cavgx(matchchunknum,numcfs(matchchunknum))
      floatavgy(1)=cavgy(matchchunknum,numcfs(matchchunknum))
      floatstdx(1)=cstdx(matchchunknum,numcfs(matchchunknum))
      floatstdy(1)=cstdy(matchchunknum,numcfs(matchchunknum))
      floatmaxx(1)=cmaxx(matchchunknum,numcfs(matchchunknum))
      floatmaxy(1)=cmaxy(matchchunknum,numcfs(matchchunknum))
      floatminx(1)=cminx(matchchunknum,numcfs(matchchunknum))
      floatminy(1)=cminy(matchchunknum,numcfs(matchchunknum))
      floatnobif(1)=.TRUE.
      floatfsfromslice(1)=cfsfromslice(matchchunknum,numcfs(matchchunknum))
      carea(matchchunknum,numcfs(matchchunknum))=slfsarea(slice,fs)
      cmstar(matchchunknum,numcfs(matchchunknum))=slfsmstar(slice,fs)
      cfreq(matchchunknum,numcfs(matchchunknum))=slfsfreq(slice,fs)
      corbtype(matchchunknum,numcfs(matchchunknum))=slfsorbtype(slice,fs)
      corbnum(matchchunknum,numcfs(matchchunknum))=slfsorbnum(slice,fs)
      cavgx(matchchunknum,numcfs(matchchunknum))=slfsavgx(slice,fs)
      cavgy(matchchunknum,numcfs(matchchunknum))=slfsavgy(slice,fs)
      cstdx(matchchunknum,numcfs(matchchunknum))=slfsstdx(slice,fs)
      cstdy(matchchunknum,numcfs(matchchunknum))=slfsstdy(slice,fs)
      cmaxx(matchchunknum,numcfs(matchchunknum))=slfsmaxx(slice,fs)
      cmaxy(matchchunknum,numcfs(matchchunknum))=slfsmaxy(slice,fs)
      cminx(matchchunknum,numcfs(matchchunknum))=slfsminx(slice,fs)
      cminy(matchchunknum,numcfs(matchchunknum))=slfsminy(slice,fs)
      cnobif(matchchunknum,numcfs(matchchunknum))=.TRUE.
      cfsfromslice(matchchunknum,numcfs(matchchunknum))=slice
      badnessoffit(matchchunknum,numcfs(matchchunknum))=oldbadness
      nonextarea(matchchunknum,numcfs(matchchunknum))=.TRUE.
      noprevarea(matchchunknum,numcfs(matchchunknum))=.FALSE.
      nonextarea(matchchunknum,numcfs(matchchunknum)-1)=.FALSE.
      wehaveafloater=.TRUE.
      floatloopcount=0

      do
 !************** start floater loop
       floatloopcount=floatloopcount+1
    matchchunknum=0
    matchchunkocc=.FALSE.
    oldbadness=1000000.0D0
    do chunk=1,numchunks
     if ((cfsfromslice(chunk,numcfs(chunk))==floatfsfromslice(1)).AND.(numcfs(chunk)>1)) then
      mcond(1)=(floatavgx(1)<cavgx(chunk,numcfs(chunk)-1)+1.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(floatavgx(1)>cavgx(chunk,numcfs(chunk)-1)-1.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(2)=(floatavgy(1)<cavgy(chunk,numcfs(chunk)-1)+1.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(floatavgy(1)>cavgy(chunk,numcfs(chunk)-1)-1.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(3)=(floatmaxx(1)<cmaxx(chunk,numcfs(chunk)-1)+2.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(floatmaxx(1)>cmaxx(chunk,numcfs(chunk)-1)-2.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(4)=(floatmaxy(1)<cmaxy(chunk,numcfs(chunk)-1)+2.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(floatmaxy(1)>cmaxy(chunk,numcfs(chunk)-1)-2.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(5)=(floatminx(1)<cminx(chunk,numcfs(chunk)-1)+2.0D0*cstdx(chunk,numcfs(chunk)-1)) &
&.AND.(floatminx(1)>cminx(chunk,numcfs(chunk)-1)-2.0D0*cstdx(chunk,numcfs(chunk)-1))
      mcond(6)=(floatminy(1)<cminy(chunk,numcfs(chunk)-1)+2.0D0*cstdy(chunk,numcfs(chunk)-1)) &
&.AND.(floatminy(1)>cminy(chunk,numcfs(chunk)-1)-2.0D0*cstdy(chunk,numcfs(chunk)-1))
      mcond(7)=(slfsorbtype(slice,fs)==corbtype(chunk,numcfs(chunk)-1))
      mcond(8)=(cavgx(chunk,numcfs(chunk)-1)<floatavgx(1)+1.0D0*floatstdx(1)) &
&.AND.(cavgx(chunk,numcfs(chunk)-1)>floatavgx(1)-1.0D0*floatstdx(1))
      mcond(9)=(cavgy(chunk,numcfs(chunk)-1)<floatavgy(1)+1.0D0*floatstdy(1)) &
&.AND.(cavgy(chunk,numcfs(chunk)-1)>floatavgy(1)-1.0D0*floatstdy(1))
      mcond(10)=(cmaxx(chunk,numcfs(chunk)-1)<floatmaxx(1)+2.0D0*floatstdx(1)) &
&.AND.(cmaxx(chunk,numcfs(chunk)-1)>floatmaxx(1)-2.0D0*floatstdx(1))
      mcond(11)=(cmaxy(chunk,numcfs(chunk)-1)<floatmaxy(1)+2.0D0*floatstdy(1)) &
&.AND.(cmaxy(chunk,numcfs(chunk)-1)>floatmaxy(1)-2.0D0*floatstdy(1))
      mcond(12)=(cminx(chunk,numcfs(chunk)-1)<floatminx(1)+2.0D0*floatstdx(1)) &
&.AND.(cminx(chunk,numcfs(chunk)-1)>floatminx(1)-2.0D0*floatstdx(1))
      mcond(13)=(cminy(chunk,numcfs(chunk)-1)<floatminy(1)+2.0D0*floatstdy(1)) &
&.AND.(cminy(chunk,numcfs(chunk)-1)>floatminy(1)-2.0D0*floatstdy(1))
      mcond(14)=(floatavgx(1)<cmaxx(chunk,numcfs(chunk)-1)) &
&.AND.(floatavgx(1)>cminx(chunk,numcfs(chunk)-1))
      mcond(15)=(floatavgy(1)<cmaxy(chunk,numcfs(chunk)-1)) &
&.AND.(floatavgy(1)>cminy(chunk,numcfs(chunk)-1))
      mcond(16)=(cavgx(chunk,numcfs(chunk)-1)<floatmaxx(1)) &
&.AND.(cavgx(chunk,numcfs(chunk)-1)>floatminx(1))
      mcond(17)=(cavgy(chunk,numcfs(chunk)-1)<floatmaxy(1)) &
&.AND.(cavgy(chunk,numcfs(chunk)-1)>floatminy(1))
      mcond(18)=floatnobif(1)
      mcond(19)=cnobif(chunk,numcfs(chunk)-1)
      if (mcond(1).AND.mcond(2).AND.mcond(3).AND.mcond(4).AND.mcond(5).AND.mcond(6).AND.mcond(7) &
&.AND.mcond(8).AND.mcond(9).AND.mcond(10).AND.mcond(11).AND.mcond(12).AND.mcond(13) &
&.AND.mcond(14).AND.mcond(15).AND.mcond(16).AND.mcond(17).AND.mcond(18).AND.mcond(19)) then
       badness=((cavgx(chunk,numcfs(chunk)-1))-(floatavgx(1)))* &
&((cavgx(chunk,numcfs(chunk)-1))-(floatavgx(1)))
       badness=badness+(((cavgy(chunk,numcfs(chunk)-1))-(floatavgy(1)))* &
&((cavgy(chunk,numcfs(chunk)-1))-(floatavgy(1))))
       badness=badness+(((cmaxx(chunk,numcfs(chunk)-1))-(floatmaxx(1)))* &
&((cmaxx(chunk,numcfs(chunk)-1))-(floatmaxx(1))))
       badness=badness+(((cmaxy(chunk,numcfs(chunk)-1))-(floatmaxy(1)))* &
&((cmaxy(chunk,numcfs(chunk)-1))-(floatmaxy(1))))
       badness=badness+(((cminx(chunk,numcfs(chunk)-1))-(floatminx(1)))* &
&((cminx(chunk,numcfs(chunk)-1))-(floatminx(1))))
       badness=badness+(((cminy(chunk,numcfs(chunk)-1))-(floatminy(1)))* &
&((cminy(chunk,numcfs(chunk)-1))-(floatminy(1))))
       if ((badness<badnessoffit(chunk,numcfs(chunk))).AND.(badness<oldbadness)) then
        oldbadness=badness
        matchchunkocc=.TRUE.
        matchchunknum=chunk
       end if
      end if
     else
      if ((cfsfromslice(chunk,numcfs(chunk)))==(floatfsfromslice(1)-1)) then
       mcond(1)=(floatavgx(1)<cavgx(chunk,numcfs(chunk))+1.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(floatavgx(1)>cavgx(chunk,numcfs(chunk))-1.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(2)=(floatavgy(1)<cavgy(chunk,numcfs(chunk))+1.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(floatavgy(1)>cavgy(chunk,numcfs(chunk))-1.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(3)=(floatmaxx(1)<cmaxx(chunk,numcfs(chunk))+2.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(floatmaxx(1)>cmaxx(chunk,numcfs(chunk))-2.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(4)=(floatmaxy(1)<cmaxy(chunk,numcfs(chunk))+2.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(floatmaxy(1)>cmaxy(chunk,numcfs(chunk))-2.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(5)=(floatminx(1)<cminx(chunk,numcfs(chunk))+2.0D0*cstdx(chunk,numcfs(chunk))) &
&.AND.(floatminx(1)>cminx(chunk,numcfs(chunk))-2.0D0*cstdx(chunk,numcfs(chunk)))
       mcond(6)=(floatminy(1)<cminy(chunk,numcfs(chunk))+2.0D0*cstdy(chunk,numcfs(chunk))) &
&.AND.(floatminy(1)>cminy(chunk,numcfs(chunk))-2.0D0*cstdy(chunk,numcfs(chunk)))
       mcond(7)=(slfsorbtype(slice,fs)==corbtype(chunk,numcfs(chunk)))
       mcond(8)=(cavgx(chunk,numcfs(chunk))<floatavgx(1)+1.0D0*floatstdx(1)) &
&.AND.(cavgx(chunk,numcfs(chunk))>floatavgx(1)-1.0D0*floatstdx(1))
       mcond(9)=(cavgy(chunk,numcfs(chunk))<floatavgy(1)+1.0D0*floatstdy(1)) &
&.AND.(cavgy(chunk,numcfs(chunk))>floatavgy(1)-1.0D0*floatstdy(1))
       mcond(10)=(cmaxx(chunk,numcfs(chunk))<floatmaxx(1)+2.0D0*floatstdx(1)) &
&.AND.(cmaxx(chunk,numcfs(chunk))>floatmaxx(1)-2.0D0*floatstdx(1))
       mcond(11)=(cmaxy(chunk,numcfs(chunk))<floatmaxy(1)+2.0D0*floatstdy(1)) &
&.AND.(cmaxy(chunk,numcfs(chunk))>floatmaxy(1)-2.0D0*floatstdy(1))
       mcond(12)=(cminx(chunk,numcfs(chunk))<floatminx(1)+2.0D0*floatstdx(1)) &
&.AND.(cminx(chunk,numcfs(chunk))>floatminx(1)-2.0D0*floatstdx(1))
       mcond(13)=(cminy(chunk,numcfs(chunk))<floatminy(1)+2.0D0*floatstdy(1)) &
&.AND.(cminy(chunk,numcfs(chunk))>floatminy(1)-2.0D0*floatstdy(1))
       mcond(14)=(floatavgx(1)<cmaxx(chunk,numcfs(chunk))) &
&.AND.(floatavgx(1)>cminx(chunk,numcfs(chunk)))
       mcond(15)=(floatavgy(1)<cmaxy(chunk,numcfs(chunk))) &
&.AND.(floatavgy(1)>cminy(chunk,numcfs(chunk)))
       mcond(16)=(cavgx(chunk,numcfs(chunk))<floatmaxx(1)) &
&.AND.(cavgx(chunk,numcfs(chunk))>floatminx(1))
       mcond(17)=(cavgy(chunk,numcfs(chunk))<floatmaxy(1)) &
&.AND.(cavgy(chunk,numcfs(chunk))>floatminy(1))
       mcond(18)=floatnobif(1)
       mcond(19)=cnobif(chunk,numcfs(chunk))
       if (mcond(1).AND.mcond(2).AND.mcond(3).AND.mcond(4).AND.mcond(5).AND.mcond(6).AND.mcond(7) &
&.AND.mcond(8).AND.mcond(9).AND.mcond(10).AND.mcond(11).AND.mcond(12).AND.mcond(13) &
&.AND.mcond(14).AND.mcond(15).AND.mcond(16).AND.mcond(17).AND.mcond(18).AND.mcond(19)) then
        badness=((cavgx(chunk,numcfs(chunk)))-(floatavgx(1))) &
&*((cavgx(chunk,numcfs(chunk)))-(floatavgx(1)))
        badness=badness+(((cavgy(chunk,numcfs(chunk)))-(floatavgy(1)))* &
&((cavgy(chunk,numcfs(chunk)))-(floatavgy(1))))
        badness=badness+(((cmaxx(chunk,numcfs(chunk)))-(floatmaxx(1)))* &
&((cmaxx(chunk,numcfs(chunk)))-(floatmaxx(1))))
        badness=badness+(((cmaxy(chunk,numcfs(chunk)))-(floatmaxy(1)))* &
&((cmaxy(chunk,numcfs(chunk)))-(floatmaxy(1))))
        badness=badness+(((cminx(chunk,numcfs(chunk)))-(floatminx(1)))* &
&((cminx(chunk,numcfs(chunk)))-(floatminx(1))))
        badness=badness+(((cminy(chunk,numcfs(chunk)))-(floatminy(1)))* &
&((cminy(chunk,numcfs(chunk)))-(floatminy(1))))
        if (badness<oldbadness) then
         oldbadness=badness
         matchchunkocc=.FALSE.
         matchchunknum=chunk
        end if
       end if
      end if
     end if
    end do

       if (matchchunknum==0) then
        numchunks=numchunks+1
        numcfs(numchunks)=1
        noprevarea(numchunks,1)=.TRUE.
        nonextarea(numchunks,1)=.TRUE.
        carea(numchunks,1)=floatarea(1)
        cmstar(numchunks,1)=floatmstar(1)
        cfreq(numchunks,1)=floatfreq(1)
        corbtype(numchunks,1)=floatorbtype(1)
        corbnum(numchunks,1)=floatorbnum(1)
        cavgx(numchunks,1)=floatavgx(1)
        cavgy(numchunks,1)=floatavgy(1)
        cstdx(numchunks,1)=floatstdx(1)
        cstdy(numchunks,1)=floatstdy(1)
        cmaxx(numchunks,1)=floatmaxx(1)
        cmaxy(numchunks,1)=floatmaxy(1)
        cminx(numchunks,1)=floatminx(1)
        cminy(numchunks,1)=floatminy(1)
        cnobif(numchunks,1)=.TRUE.
        cfsfromslice(numchunks,1)=floatfsfromslice(1)
        badnessoffit(numchunks,1)=0.0D0
        wehaveafloater=.FALSE.
       else
        if (matchchunkocc.eqv..FALSE.) then
         numcfs(matchchunknum)=numcfs(matchchunknum)+1
         carea(matchchunknum,numcfs(matchchunknum))=floatarea(1)
         cmstar(matchchunknum,numcfs(matchchunknum))=floatmstar(1)
         cfreq(matchchunknum,numcfs(matchchunknum))=floatfreq(1)
         corbtype(matchchunknum,numcfs(matchchunknum))=floatorbtype(1)
         corbnum(matchchunknum,numcfs(matchchunknum))=floatorbnum(1)
         cavgx(matchchunknum,numcfs(matchchunknum))=floatavgx(1)
         cavgy(matchchunknum,numcfs(matchchunknum))=floatavgy(1)
         cstdx(matchchunknum,numcfs(matchchunknum))=floatstdx(1)
         cstdy(matchchunknum,numcfs(matchchunknum))=floatstdy(1)
         cmaxx(matchchunknum,numcfs(matchchunknum))=floatmaxx(1)
         cmaxy(matchchunknum,numcfs(matchchunknum))=floatmaxy(1)
         cminx(matchchunknum,numcfs(matchchunknum))=floatminx(1)
         cminy(matchchunknum,numcfs(matchchunknum))=floatminy(1)
         cnobif(matchchunknum,numcfs(matchchunknum))=.TRUE.
         cfsfromslice(matchchunknum,numcfs(matchchunknum))=floatfsfromslice(1)
         badnessoffit(matchchunknum,numcfs(matchchunknum))=oldbadness
         nonextarea(matchchunknum,numcfs(matchchunknum))=.TRUE.
         noprevarea(matchchunknum,numcfs(matchchunknum))=.FALSE.
         nonextarea(matchchunknum,numcfs(matchchunknum)-1)=.FALSE.
         wehaveafloater=.FALSE.
        else
         floatarea(2)=carea(matchchunknum,numcfs(matchchunknum))
         floatmstar(2)=cmstar(matchchunknum,numcfs(matchchunknum))
         floatfreq(2)=cfreq(matchchunknum,numcfs(matchchunknum))
         floatorbtype(2)=corbtype(matchchunknum,numcfs(matchchunknum))
         floatorbnum(2)=corbnum(matchchunknum,numcfs(matchchunknum))
         floatavgx(2)=cavgx(matchchunknum,numcfs(matchchunknum))
         floatavgy(2)=cavgy(matchchunknum,numcfs(matchchunknum))
         floatstdx(2)=cstdx(matchchunknum,numcfs(matchchunknum))
         floatstdy(2)=cstdy(matchchunknum,numcfs(matchchunknum))
         floatmaxx(2)=cmaxx(matchchunknum,numcfs(matchchunknum))
         floatmaxy(2)=cmaxy(matchchunknum,numcfs(matchchunknum))
         floatminx(2)=cminx(matchchunknum,numcfs(matchchunknum))
         floatminy(2)=cminy(matchchunknum,numcfs(matchchunknum))
         floatnobif(2)=.TRUE.
         floatfsfromslice(2)=cfsfromslice(matchchunknum,numcfs(matchchunknum))
         carea(matchchunknum,numcfs(matchchunknum))=floatarea(1)
         cmstar(matchchunknum,numcfs(matchchunknum))=floatmstar(1)
         cfreq(matchchunknum,numcfs(matchchunknum))=floatfreq(1)
         corbtype(matchchunknum,numcfs(matchchunknum))=floatorbtype(1)
         corbnum(matchchunknum,numcfs(matchchunknum))=floatorbnum(1)
         cavgx(matchchunknum,numcfs(matchchunknum))=floatavgx(1)
         cavgy(matchchunknum,numcfs(matchchunknum))=floatavgy(1)
         cstdx(matchchunknum,numcfs(matchchunknum))=floatstdx(1)
         cstdy(matchchunknum,numcfs(matchchunknum))=floatstdy(1)
         cmaxx(matchchunknum,numcfs(matchchunknum))=floatmaxx(1)
         cmaxy(matchchunknum,numcfs(matchchunknum))=floatmaxy(1)
         cminx(matchchunknum,numcfs(matchchunknum))=floatminx(1)
         cminy(matchchunknum,numcfs(matchchunknum))=floatminy(1)
         cnobif(matchchunknum,numcfs(matchchunknum))=.TRUE.
         cfsfromslice(matchchunknum,numcfs(matchchunknum))=floatfsfromslice(1)
         badnessoffit(matchchunknum,numcfs(matchchunknum))=oldbadness
         nonextarea(matchchunknum,numcfs(matchchunknum))=.TRUE.
         noprevarea(matchchunknum,numcfs(matchchunknum))=.FALSE.
         nonextarea(matchchunknum,numcfs(matchchunknum)-1)=.FALSE.
         floatarea(1)=floatarea(2)
         floatmstar(1)=floatmstar(2)
         floatfreq(1)=floatfreq(2)
         floatorbtype(1)=floatorbtype(2)
         floatorbnum(1)=floatorbnum(2)
         floatavgx(1)=floatavgx(2)
         floatavgy(1)=floatavgy(2)
         floatstdx(1)=floatstdx(2)
         floatstdy(1)=floatstdy(2)
         floatmaxx(1)=floatmaxx(2)
         floatmaxy(1)=floatmaxy(2)
         floatminx(1)=floatminx(2)
         floatminy(1)=floatminy(2)
         floatfsfromslice(1)=floatfsfromslice(2)
         wehaveafloater=.TRUE.
        end if
       end if
       
       if (floatloopcount>500) then
        wehaveafloater=.FALSE.
        write(*,'('' Float loop got stuck. Do not know why. '')')
        write(17,'('' Float loop got stuck. Do not know why. '')')
       end if

       if (wehaveafloater.eqv..FALSE.) exit
 !************** end floater loop
      end do
 !************** end of special floater section
     end if
    end if

   end if

  end do
 end do

 write(*,'('' Finding extremal areas... '')')

 numsorted=0
 numnd=0
 numtemp=0
 newnumnd=0

 !EXTREMAL AREA FINDER

 do chunk=1,numchunks
  if (hvd/='r') then
   write(17,'('' Chunk '',I5,'' has '',I5,'' slices'')')chunk,numcfs(chunk)
  end if
  do cfs=1,numcfs(chunk)
   isextremal=.FALSE.
   mcond(33)=(cmaxx(chunk,cfs)<(1.0D0-2.0D0*cstdx(chunk,cfs)))
   mcond(34)=(cmaxy(chunk,cfs)<(1.0D0-2.0D0*cstdy(chunk,cfs)))
   mcond(35)=(cminx(chunk,cfs)>(0.0D0+2.0D0*cstdx(chunk,cfs)))
   mcond(36)=(cminy(chunk,cfs)>(0.0D0+2.0D0*cstdy(chunk,cfs)))
   mcond(37)=(mcond(33).AND.mcond(34).AND.mcond(35).AND.mcond(36))
   if ((cfreq(chunk,cfs)>minextfreq).AND.(mcond(37).OR.(allowextnearwalls=='y'))) then
    if ((noprevarea(chunk,cfs).eqv..FALSE.).AND.(nonextarea(chunk,cfs).eqv..FALSE.)) then
     if ((carea(chunk,(cfs-1))<carea(chunk,cfs)).AND.(carea(chunk,(cfs+1))<carea(chunk,cfs))) then
      isextremal=.TRUE.
     end if
     if ((carea(chunk,(cfs-1))>carea(chunk,cfs)).AND.(carea(chunk,(cfs+1))>carea(chunk,cfs))) then
      isextremal=.TRUE.
     end if
     if (carea(chunk,(cfs-1))==carea(chunk,cfs)) then
      isextremal=.TRUE.
     end if
    else
     isextremal=.FALSE.
    end if
    if (isextremal.eqv..TRUE.) then
      ndcurvarray(numnd+1)=convfsarea2kt*((carea(chunk,(cfs-1))+carea(chunk,(cfs+1))-&
&2.0D0*carea(chunk,cfs))/(intkpointspacing*intkpointspacing))
     if (hvd/='r') then
      write(17,'('' '')')
      if (corbtype(chunk,cfs)==1.0D0) then
       write(17,'('' Extremal electron orbit: orbit '',I5,'' on slice '',I4)')corbnum(chunk,cfs),cfsfromslice(chunk,cfs)
      end if
      if (corbtype(chunk,cfs)==-1.0D0) then
       write(17,'('' Extremal hole orbit: orbit '',I5,'' on slice '',I4)')corbnum(chunk,cfs),cfsfromslice(chunk,cfs)
      end if
      write(17,'('' Located on FS chunk '',I5)')chunk
      write(17,'('' Area = '',F8.4,'' A^(-2), Freq. = '',F8.4,'' kT, m* = '',F8.4 &
&,'' m_e, Curv. = '',ES12.4,'' kT A^2'')')carea(chunk,cfs),cfreq(chunk,cfs),cmstar(chunk,cfs),ndcurvarray(numnd+1)
      write(17,'('' Average (x,y) = ('',F6.3,'' +/- '',F6.3,'', '',F6.3,'' +/- '',F6.3 &
&,'') in SC fractional coordinates'')')cavgx(chunk,cfs),cstdx(chunk,cfs),cavgy(chunk,cfs),cstdy(chunk,cfs)
      write(17,'('' maxx = '',F6.3,'', maxy = '',F6.3,'', minx = '',F6.3 &
&,'', miny = '',F6.3)')cmaxx(chunk,cfs),cmaxy(chunk,cfs),cminx(chunk,cfs),cminy(chunk,cfs)
     end if

     ndfreqarray(numnd+1)=cfreq(chunk,cfs)
     ndmassarray(numnd+1)=cmstar(chunk,cfs)
     ndorbtypearray(numnd+1)=corbtype(chunk,cfs)
     ndorbnumarray(numnd+1)=corbnum(chunk,cfs)
     ndfromslicearray(numnd+1)=cfsfromslice(chunk,cfs)

     cp1=maxlreciplat*((((4.0D0*cavgx(chunk,cfs))-1.0D0)*(p*p*u + c))-(((4.0D0*cavgy(chunk,cfs)) &
&-1.0D0)*(p*q*u))+(((4.0D0*(dble(cfsfromslice(chunk,cfs))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(q*s)))
     cp2=maxlreciplat*((((4.0D0*cavgx(chunk,cfs))-1.0D0)*(-p*q*u))+(((4.0D0*cavgy(chunk,cfs)) &
&-1.0D0)*(q*q*u + c))+(((4.0D0*(dble(cfsfromslice(chunk,cfs))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(p*s)))
     cp3=maxlreciplat*((((4.0D0*cavgx(chunk,cfs))-1.0D0)*(-q*s))-(((4.0D0*cavgy(chunk,cfs)) &
&-1.0D0)*(p*s))+(((4.0D0*(dble(cfsfromslice(chunk,cfs))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(c)))

     cfintpointx=((ai*cp1)-(aii*cp2)+(aiii*cp3))/bigd
     cfintpointy=((bi*cp1)-(bii*cp2)+(biii*cp3))/bigd
     cfintpointz=((ci*cp1)-(cii*cp2)+(ciii*cp3))/bigd

     csubtractorx=floor(cfintpointx)
     csubtractory=floor(cfintpointy)
     csubtractorz=floor(cfintpointz)

     ndavgxrucarray(numnd+1)=cfintpointx-csubtractorx
     ndavgyrucarray(numnd+1)=cfintpointy-csubtractory
     ndavgzrucarray(numnd+1)=cfintpointz-csubtractorz

     numnd=numnd+1

    end if
   end if
  end do
 end do

 numleftinarray=numnd
 numav2=0

 do while (numleftinarray>0)

  numsamecentre=0
  numdiffcentre=0

  do cnsearchcounter=1,numleftinarray
   ! if same centre (or 1st iteration), then put into ndcn arrays and increment numsamecentre by 1; if not, then put into cntemp arrays
   if (cnsearchcounter==1) then
    numsamecentre=numsamecentre+1
    ndcnfreqarray(numsamecentre)=ndfreqarray(cnsearchcounter)
    ndcnmassarray(numsamecentre)=ndmassarray(cnsearchcounter)
    ndcncurvarray(numsamecentre)=ndcurvarray(cnsearchcounter)
    ndcnorbtypearray(numsamecentre)=ndorbtypearray(cnsearchcounter)
    ndcnorbnumarray(numsamecentre)=ndorbnumarray(cnsearchcounter)
    ndcnfromslicearray(numsamecentre)=ndfromslicearray(cnsearchcounter)
    ndcnavgxrucarray(numsamecentre)=ndavgxrucarray(cnsearchcounter)
    ndcnavgyrucarray(numsamecentre)=ndavgyrucarray(cnsearchcounter)
    ndcnavgzrucarray(numsamecentre)=ndavgzrucarray(cnsearchcounter)
   else

    closex=(dabs((ndcnavgxrucarray(1))-(ndavgxrucarray(cnsearchcounter)))<avgsamefrac)
    closenegx=(dabs((ndcnavgxrucarray(1))-(ndavgxrucarray(cnsearchcounter)-1.0D0))<avgsamefrac)
    closeposx=(dabs((ndcnavgxrucarray(1))-(ndavgxrucarray(cnsearchcounter)+1.0D0))<avgsamefrac)
    closey=(dabs((ndcnavgyrucarray(1))-(ndavgyrucarray(cnsearchcounter)))<avgsamefrac)
    closenegy=(dabs((ndcnavgyrucarray(1))-(ndavgyrucarray(cnsearchcounter)-1.0D0))<avgsamefrac)
    closeposy=(dabs((ndcnavgyrucarray(1))-(ndavgyrucarray(cnsearchcounter)+1.0D0))<avgsamefrac)
    closez=(dabs((ndcnavgzrucarray(1))-(ndavgzrucarray(cnsearchcounter)))<avgsamefrac)
    closenegz=(dabs((ndcnavgzrucarray(1))-(ndavgzrucarray(cnsearchcounter)-1.0D0))<avgsamefrac)
    closeposz=(dabs((ndcnavgzrucarray(1))-(ndavgzrucarray(cnsearchcounter)+1.0D0))<avgsamefrac)

    if (((closex).OR.(closenegx).OR.(closeposx)).AND.((closey).OR.(closenegy) &
&.OR.(closeposy)).AND.((closez).OR.(closenegz).OR.(closeposz))) then

     numsamecentre=numsamecentre+1
     ndcnfreqarray(numsamecentre)=ndfreqarray(cnsearchcounter)
     ndcnmassarray(numsamecentre)=ndmassarray(cnsearchcounter)
     ndcncurvarray(numsamecentre)=ndcurvarray(cnsearchcounter)
     ndcnorbtypearray(numsamecentre)=ndorbtypearray(cnsearchcounter)
     ndcnorbnumarray(numsamecentre)=ndorbnumarray(cnsearchcounter)
     ndcnfromslicearray(numsamecentre)=ndfromslicearray(cnsearchcounter)
     if (closex) then
      ndcnavgxrucarray(numsamecentre)=ndavgxrucarray(cnsearchcounter)
     elseif (closenegx) then
      ndcnavgxrucarray(numsamecentre)=ndavgxrucarray(cnsearchcounter)-1.0D0
     elseif (closeposx) then
      ndcnavgxrucarray(numsamecentre)=ndavgxrucarray(cnsearchcounter)+1.0D0
     end if
     if (closey) then
      ndcnavgyrucarray(numsamecentre)=ndavgyrucarray(cnsearchcounter)
     elseif (closenegy) then
      ndcnavgyrucarray(numsamecentre)=ndavgyrucarray(cnsearchcounter)-1.0D0
     elseif (closeposy) then
      ndcnavgyrucarray(numsamecentre)=ndavgyrucarray(cnsearchcounter)+1.0D0
     end if
     if (closez) then
      ndcnavgzrucarray(numsamecentre)=ndavgzrucarray(cnsearchcounter)
     elseif (closenegz) then
      ndcnavgzrucarray(numsamecentre)=ndavgzrucarray(cnsearchcounter)-1.0D0
     elseif (closeposz) then
      ndcnavgzrucarray(numsamecentre)=ndavgzrucarray(cnsearchcounter)+1.0D0
     end if

    else
     numdiffcentre=numdiffcentre+1
     cntempfreqarray(numdiffcentre)=ndfreqarray(cnsearchcounter)
     cntempmassarray(numdiffcentre)=ndmassarray(cnsearchcounter)
     cntempcurvarray(numdiffcentre)=ndcurvarray(cnsearchcounter)
     cntemporbtypearray(numdiffcentre)=ndorbtypearray(cnsearchcounter)
     cntemporbnumarray(numdiffcentre)=ndorbnumarray(cnsearchcounter)
     cntempfromslicearray(numdiffcentre)=ndfromslicearray(cnsearchcounter)
     cntempavgxrucarray(numdiffcentre)=ndavgxrucarray(cnsearchcounter)
     cntempavgyrucarray(numdiffcentre)=ndavgyrucarray(cnsearchcounter)
     cntempavgzrucarray(numdiffcentre)=ndavgzrucarray(cnsearchcounter)
    end if

   end if
  end do

  numleftinarray=numdiffcentre

  do cnsearchcounter=1,numleftinarray
   ndfreqarray(cnsearchcounter)=cntempfreqarray(cnsearchcounter)
   ndmassarray(cnsearchcounter)=cntempmassarray(cnsearchcounter)
   ndcurvarray(cnsearchcounter)=cntempcurvarray(cnsearchcounter)
   ndorbtypearray(cnsearchcounter)=cntemporbtypearray(cnsearchcounter)
   ndorbnumarray(cnsearchcounter)=cntemporbnumarray(cnsearchcounter)
   ndfromslicearray(cnsearchcounter)=cntempfromslicearray(cnsearchcounter)
   ndavgxrucarray(cnsearchcounter)=cntempavgxrucarray(cnsearchcounter)
   ndavgyrucarray(cnsearchcounter)=cntempavgyrucarray(cnsearchcounter)
   ndavgzrucarray(cnsearchcounter)=cntempavgzrucarray(cnsearchcounter)
  end do

  !-----------------------
  ! Sort unduplicated extremal freqs with similar centres

  newnumnd=numsamecentre
  numsorted=0

  do i=1,numsamecentre
   ndcnminfreq=minval(ndcnfreqarray(1:newnumnd))
   foundamin=.FALSE.
   numtemp=0
   do j=1,newnumnd
    if ((foundamin.eqv..FALSE.).AND.(ndcnfreqarray(j)==ndcnminfreq)) then
     numsorted=numsorted+1
     sortedfreqarray(numsorted)=ndcnfreqarray(j)
     sortedmassarray(numsorted)=ndcnmassarray(j)
     sortedcurvarray(numsorted)=ndcncurvarray(j)
     sortedorbtypearray(numsorted)=ndcnorbtypearray(j)
     sortedorbnumarray(numsorted)=ndcnorbnumarray(j)
     sortedfromslicearray(numsorted)=ndcnfromslicearray(j)
     sortedavgxrucarray(numsorted)=ndcnavgxrucarray(j)
     sortedavgyrucarray(numsorted)=ndcnavgyrucarray(j)
     sortedavgzrucarray(numsorted)=ndcnavgzrucarray(j)
     foundamin=.TRUE.
    else
     numtemp=numtemp+1
     tempfreqarray(numtemp)=ndcnfreqarray(j)
     tempmassarray(numtemp)=ndcnmassarray(j)
     tempcurvarray(numtemp)=ndcncurvarray(j)
     temporbtypearray(numtemp)=ndcnorbtypearray(j)
     temporbnumarray(numtemp)=ndcnorbnumarray(j)
     tempfromslicearray(numtemp)=ndcnfromslicearray(j)
     tempavgxrucarray(numtemp)=ndcnavgxrucarray(j)
     tempavgyrucarray(numtemp)=ndcnavgyrucarray(j)
     tempavgzrucarray(numtemp)=ndcnavgzrucarray(j)
    end if
   end do
   do j=1,numtemp
    ndcnfreqarray(j)=tempfreqarray(j)
    ndcnmassarray(j)=tempmassarray(j)
    ndcncurvarray(j)=tempcurvarray(j)
    ndcnorbtypearray(j)=temporbtypearray(j)
    ndcnorbnumarray(j)=temporbnumarray(j)
    ndcnfromslicearray(j)=tempfromslicearray(j)
    ndcnavgxrucarray(j)=tempavgxrucarray(j)
    ndcnavgyrucarray(j)=tempavgyrucarray(j)
    ndcnavgzrucarray(j)=tempavgzrucarray(j)
   end do
   newnumnd=numtemp
  end do

  ! Average sorted frequencies with similar centres that are within user specified range (default 1%) of one another and find sample stdev

  numaveraged=0
  numtemp=0

  do i=1,numsorted
   if (i==1) then
    numaveraged=1
    numtemp=1
    tempfreqarray(numtemp)=sortedfreqarray(i)
    tempmassarray(numtemp)=sortedmassarray(i)
    tempcurvarray(numtemp)=sortedcurvarray(i)
    temporbtypearray(numtemp)=sortedorbtypearray(i)
    temporbnumarray(numtemp)=sortedorbnumarray(i)
    tempfromslicearray(numtemp)=sortedfromslicearray(i)
    tempavgxrucarray(numtemp)=sortedavgxrucarray(i)
    tempavgyrucarray(numtemp)=sortedavgyrucarray(i)
    tempavgzrucarray(numtemp)=sortedavgzrucarray(i)
   else
    if (sortedfreqarray(i)<=((1.0+freqsamefrac)*tempfreqarray(numtemp))) then
     numtemp=numtemp+1
     tempfreqarray(numtemp)=sortedfreqarray(i)
     tempmassarray(numtemp)=sortedmassarray(i)
     tempcurvarray(numtemp)=sortedcurvarray(i)
     temporbtypearray(numtemp)=sortedorbtypearray(i)
     temporbnumarray(numtemp)=sortedorbnumarray(i)
     tempfromslicearray(numtemp)=sortedfromslicearray(i)
     tempavgxrucarray(numtemp)=sortedavgxrucarray(i)
     tempavgyrucarray(numtemp)=sortedavgyrucarray(i)
     tempavgzrucarray(numtemp)=sortedavgzrucarray(i)
    else
     freqavgsum=0.0D0
     massavgsum=0.0D0
     curvavgsum=0.0D0
     orbtypeavgsum=0.0D0
     avgxrucavgsum=0.0D0
     avgyrucavgsum=0.0D0
     avgzrucavgsum=0.0D0
     freqstdsum=0.0D0
     massstdsum=0.0D0
     curvstdsum=0.0D0
     orbtypestdsum=0.0D0
     avgxrucstdsum=0.0D0
     avgyrucstdsum=0.0D0
     avgzrucstdsum=0.0D0
     do j=1,numtemp
      freqavgsum=freqavgsum+tempfreqarray(j)
      massavgsum=massavgsum+tempmassarray(j)
      curvavgsum=curvavgsum+tempcurvarray(j)
      orbtypeavgsum=orbtypeavgsum+temporbtypearray(j)
      avgxrucavgsum=avgxrucavgsum+tempavgxrucarray(j)
      avgyrucavgsum=avgyrucavgsum+tempavgyrucarray(j)
      avgzrucavgsum=avgzrucavgsum+tempavgzrucarray(j)
     end do
     averagedfreqarray(numaveraged)=freqavgsum/dble(numtemp)
     averagedmassarray(numaveraged)=massavgsum/dble(numtemp)
     averagedcurvarray(numaveraged)=curvavgsum/dble(numtemp)
     averagedorbtypearray(numaveraged)=orbtypeavgsum/dble(numtemp)
     averagedorbnumarray(numaveraged)=temporbnumarray(numtemp) ! from the largest orbit in the averaged set
     averagedfromslicearray(numaveraged)=tempfromslicearray(numtemp) ! from the largest orbit in the averaged set
     averagedavgxrucarray(numaveraged)=avgxrucavgsum/dble(numtemp)
     averagedavgyrucarray(numaveraged)=avgyrucavgsum/dble(numtemp)
     averagedavgzrucarray(numaveraged)=avgzrucavgsum/dble(numtemp)
     do j=1,numtemp
      freqstdsum=freqstdsum+((tempfreqarray(j)-averagedfreqarray(numaveraged)) &
&*(tempfreqarray(j)-averagedfreqarray(numaveraged)))
      massstdsum=massstdsum+((tempmassarray(j)-averagedmassarray(numaveraged)) &
&*(tempmassarray(j)-averagedmassarray(numaveraged)))
      curvstdsum=curvstdsum+((tempcurvarray(j)-averagedcurvarray(numaveraged)) &
&*(tempcurvarray(j)-averagedcurvarray(numaveraged)))
      orbtypestdsum=orbtypestdsum+((temporbtypearray(j)- &
&averagedorbtypearray(numaveraged))*(temporbtypearray(j)-averagedorbtypearray(numaveraged)))
      avgxrucstdsum=avgxrucstdsum+((tempavgxrucarray(j)- &
&averagedavgxrucarray(numaveraged))*(tempavgxrucarray(j)-averagedavgxrucarray(numaveraged)))
      avgyrucstdsum=avgyrucstdsum+((tempavgyrucarray(j)- &
&averagedavgyrucarray(numaveraged))*(tempavgyrucarray(j)-averagedavgyrucarray(numaveraged)))
      avgzrucstdsum=avgzrucstdsum+((tempavgzrucarray(j)- &
&averagedavgzrucarray(numaveraged))*(tempavgzrucarray(j)-averagedavgzrucarray(numaveraged)))
     end do
     stdfreqarray(numaveraged)=dsqrt(freqstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdmassarray(numaveraged)=dsqrt(massstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdcurvarray(numaveraged)=dsqrt(curvstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdorbtypearray(numaveraged)=dsqrt(orbtypestdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdavgxrucarray(numaveraged)=dsqrt(avgxrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdavgyrucarray(numaveraged)=dsqrt(avgyrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     stdavgzrucarray(numaveraged)=dsqrt(avgzrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
     numorbsarray(numaveraged)=numtemp
     numaveraged=numaveraged+1
     numtemp=1
     tempfreqarray(numtemp)=sortedfreqarray(i)
     tempmassarray(numtemp)=sortedmassarray(i)
     tempcurvarray(numtemp)=sortedcurvarray(i)
     temporbtypearray(numtemp)=sortedorbtypearray(i)
     temporbnumarray(numtemp)=sortedorbnumarray(i)
     tempfromslicearray(numtemp)=sortedfromslicearray(i)
     tempavgxrucarray(numtemp)=sortedavgxrucarray(i)
     tempavgyrucarray(numtemp)=sortedavgyrucarray(i)
     tempavgzrucarray(numtemp)=sortedavgzrucarray(i)
    end if
   end if
  end do

  freqavgsum=0.0D0
  massavgsum=0.0D0
  curvavgsum=0.0D0
  orbtypeavgsum=0.0D0
  avgxrucavgsum=0.0D0
  avgyrucavgsum=0.0D0
  avgzrucavgsum=0.0D0
  freqstdsum=0.0D0
  massstdsum=0.0D0
  curvstdsum=0.0D0
  orbtypestdsum=0.0D0
  avgxrucstdsum=0.0D0
  avgyrucstdsum=0.0D0
  avgzrucstdsum=0.0D0
  do j=1,numtemp
   freqavgsum=freqavgsum+tempfreqarray(j)
   massavgsum=massavgsum+tempmassarray(j)
   curvavgsum=curvavgsum+tempcurvarray(j)
   orbtypeavgsum=orbtypeavgsum+temporbtypearray(j)
   avgxrucavgsum=avgxrucavgsum+tempavgxrucarray(j)
   avgyrucavgsum=avgyrucavgsum+tempavgyrucarray(j)
   avgzrucavgsum=avgzrucavgsum+tempavgzrucarray(j)
  end do
  averagedfreqarray(numaveraged)=freqavgsum/dble(numtemp)
  averagedmassarray(numaveraged)=massavgsum/dble(numtemp)
  averagedcurvarray(numaveraged)=curvavgsum/dble(numtemp)
  averagedorbtypearray(numaveraged)=orbtypeavgsum/dble(numtemp)
  averagedorbnumarray(numaveraged)=temporbnumarray(numtemp) ! from the largest orbit in the averaged set
  averagedfromslicearray(numaveraged)=tempfromslicearray(numtemp) ! from the largest orbit in the averaged set
  averagedavgxrucarray(numaveraged)=avgxrucavgsum/dble(numtemp)
  averagedavgyrucarray(numaveraged)=avgyrucavgsum/dble(numtemp)
  averagedavgzrucarray(numaveraged)=avgzrucavgsum/dble(numtemp)
  do j=1,numtemp
   freqstdsum=freqstdsum+((tempfreqarray(j)-averagedfreqarray(numaveraged)) &
&*(tempfreqarray(j)-averagedfreqarray(numaveraged)))
   massstdsum=massstdsum+((tempmassarray(j)-averagedmassarray(numaveraged)) &
&*(tempmassarray(j)-averagedmassarray(numaveraged)))
   curvstdsum=curvstdsum+((tempcurvarray(j)-averagedcurvarray(numaveraged)) &
&*(tempcurvarray(j)-averagedcurvarray(numaveraged)))
   orbtypestdsum=orbtypestdsum+((temporbtypearray(j)-averagedorbtypearray(numaveraged))* &
&(temporbtypearray(j)-averagedorbtypearray(numaveraged)))
   avgxrucstdsum=avgxrucstdsum+((tempavgxrucarray(j)-averagedavgxrucarray(numaveraged))* &
&(tempavgxrucarray(j)-averagedavgxrucarray(numaveraged)))
   avgyrucstdsum=avgyrucstdsum+((tempavgyrucarray(j)-averagedavgyrucarray(numaveraged))* &
&(tempavgyrucarray(j)-averagedavgyrucarray(numaveraged)))
   avgzrucstdsum=avgzrucstdsum+((tempavgzrucarray(j)-averagedavgzrucarray(numaveraged))* &
&(tempavgzrucarray(j)-averagedavgzrucarray(numaveraged)))
  end do
  stdfreqarray(numaveraged)=dsqrt(freqstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  stdmassarray(numaveraged)=dsqrt(massstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  stdcurvarray(numaveraged)=dsqrt(dabs(curvstdsum/dble(numtemp - 1))) ! This is a "sample" standard deviation
  stdorbtypearray(numaveraged)=dsqrt(orbtypestdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  stdavgxrucarray(numaveraged)=dsqrt(avgxrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  stdavgyrucarray(numaveraged)=dsqrt(avgyrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  stdavgzrucarray(numaveraged)=dsqrt(avgzrucstdsum/dble(numtemp - 1)) ! This is a "sample" standard deviation
  numorbsarray(numaveraged)=numtemp

  !--------------------

  do j=1,numaveraged
   numav2=numav2+1
   av2freqarray(numav2)=averagedfreqarray(j)
   av2massarray(numav2)=averagedmassarray(j)
   av2curvarray(numav2)=averagedcurvarray(j)
   av2orbtypearray(numav2)=averagedorbtypearray(j)
   av2orbnumarray(numav2)=averagedorbnumarray(j)
   av2fromslicearray(numav2)=averagedfromslicearray(j)
   av2avgxrucarray(numav2)=averagedavgxrucarray(j)
   av2avgyrucarray(numav2)=averagedavgyrucarray(j)
   av2avgzrucarray(numav2)=averagedavgzrucarray(j)
   av2stdfreqarray(numav2)=stdfreqarray(j)
   av2stdmassarray(numav2)=stdmassarray(j)
   av2stdcurvarray(numav2)=stdcurvarray(j)
   av2stdorbtypearray(numav2)=stdorbtypearray(j)
   av2stdavgxrucarray(numav2)=stdavgxrucarray(j)
   av2stdavgyrucarray(numav2)=stdavgyrucarray(j)
   av2stdavgzrucarray(numav2)=stdavgzrucarray(j)
   av2numorbsarray(numav2)=numorbsarray(j)
  end do

 end do

 ! Sort final list of extremal freqs (from all centres)

 newnumnd=numav2
 numsorted=0

 do i=1,numav2
  ndcnminfreq=minval(av2freqarray(1:newnumnd))
  foundamin=.FALSE.
  numtemp=0
  do j=1,newnumnd
   if ((foundamin.eqv..FALSE.).AND.(av2freqarray(j)==ndcnminfreq)) then
    numsorted=numsorted+1
    sr2freqarray(numsorted)=av2freqarray(j)
    sr2massarray(numsorted)=av2massarray(j)
    sr2curvarray(numsorted)=av2curvarray(j)
    sr2orbtypearray(numsorted)=av2orbtypearray(j)
    sr2orbnumarray(numsorted)=av2orbnumarray(j)
    sr2fromslicearray(numsorted)=av2fromslicearray(j)
    sr2avgxrucarray(numsorted)=av2avgxrucarray(j)
    sr2avgyrucarray(numsorted)=av2avgyrucarray(j)
    sr2avgzrucarray(numsorted)=av2avgzrucarray(j)
    sr2stdfreqarray(numsorted)=av2stdfreqarray(j)
    sr2stdmassarray(numsorted)=av2stdmassarray(j)
    sr2stdcurvarray(numsorted)=av2stdcurvarray(j)
    sr2stdorbtypearray(numsorted)=av2stdorbtypearray(j)
    sr2stdavgxrucarray(numsorted)=av2stdavgxrucarray(j)
    sr2stdavgyrucarray(numsorted)=av2stdavgyrucarray(j)
    sr2stdavgzrucarray(numsorted)=av2stdavgzrucarray(j)
    sr2numorbsarray(numsorted)=av2numorbsarray(j)
    foundamin=.TRUE.
   else
    numtemp=numtemp+1
    tempfreqarray(numtemp)=av2freqarray(j)
    tempmassarray(numtemp)=av2massarray(j)
    tempcurvarray(numtemp)=av2curvarray(j)
    temporbtypearray(numtemp)=av2orbtypearray(j)
    temporbnumarray(numtemp)=av2orbnumarray(j)
    tempfromslicearray(numtemp)=av2fromslicearray(j)
    tempavgxrucarray(numtemp)=av2avgxrucarray(j)
    tempavgyrucarray(numtemp)=av2avgyrucarray(j)
    tempavgzrucarray(numtemp)=av2avgzrucarray(j)
    tempstdfreqarray(numtemp)=av2stdfreqarray(j)
    tempstdmassarray(numtemp)=av2stdmassarray(j)
    tempstdcurvarray(numtemp)=av2stdcurvarray(j)
    tempstdorbtypearray(numtemp)=av2stdorbtypearray(j)
    tempstdavgxrucarray(numtemp)=av2stdavgxrucarray(j)
    tempstdavgyrucarray(numtemp)=av2stdavgyrucarray(j)
    tempstdavgzrucarray(numtemp)=av2stdavgzrucarray(j)
    tempnumorbsarray(numtemp)=av2numorbsarray(j)
   end if
  end do
  do j=1,numtemp
   av2freqarray(j)=tempfreqarray(j)
   av2massarray(j)=tempmassarray(j)
   av2curvarray(j)=tempcurvarray(j)
   av2orbtypearray(j)=temporbtypearray(j)
   av2orbnumarray(j)=temporbnumarray(j)
   av2fromslicearray(j)=tempfromslicearray(j)
   av2avgxrucarray(j)=tempavgxrucarray(j)
   av2avgyrucarray(j)=tempavgyrucarray(j)
   av2avgzrucarray(j)=tempavgzrucarray(j)
   av2stdfreqarray(j)=tempstdfreqarray(j)
   av2stdmassarray(j)=tempstdmassarray(j)
   av2stdcurvarray(j)=tempstdcurvarray(j)
   av2stdorbtypearray(j)=tempstdorbtypearray(j)
   av2stdavgxrucarray(j)=tempstdavgxrucarray(j)
   av2stdavgyrucarray(j)=tempstdavgyrucarray(j)
   av2stdavgzrucarray(j)=tempstdavgzrucarray(j)
   av2numorbsarray(j)=tempnumorbsarray(j)
  end do
  newnumnd=numtemp
 end do

!---start orbit outline output section
 if (hvd/='r') then
  if (numsorted>0) then

   write(*,'('' Writing orbit outlines results file:   0.0 %''$)')
   percentdone=0.0D0

! write header
   write(20,'(''Theta(deg) = '',F10.6,'' , Phi(deg) = '',F10.6,'' , Number of orbits = '',I5,&
&'' , (Angstrom^-1) units'')')180.0D0*theta/pi,180.0D0*phi/pi,numsorted
   write(21,'(''Theta(deg) = '',F10.6,'' , Phi(deg) = '',F10.6,'' , Number of orbits = '',I5,&
&'' , (a.u.^-1) units'')')180.0D0*theta/pi,180.0D0*phi/pi,numsorted

! find and write orbits
   do i=1,numsorted
    CALL sliceext(sr2fromslicearray(i),sr2orbnumarray(i),sliceeleca,sliceholea)
    write(20,'(''Slice   = '',I6,'' , Freq(kT, average of all copies)              = '',F8.4)')&
&sr2fromslicearray(i),sr2freqarray(i)
    write(20,'(''Orbit # = '',I6,'' , Freq(kT, this largest copy used for outline) = '',F8.4)')&
&sr2orbnumarray(i),outlinefreq
    write(20,'(''Points  = '',I6)')noutlinepts
    write(20,'(''  kx            ky            kz'')')
    write(21,'(''Slice   = '',I6,'' , Freq(kT, average of all copies)              = '',F8.4)')&
&sr2fromslicearray(i),sr2freqarray(i)
    write(21,'(''Orbit # = '',I6,'' , Freq(kT, this largest copy used for outline) = '',F8.4)')&
&sr2orbnumarray(i),outlinefreq
    write(21,'(''Points  = '',I6)')noutlinepts
    write(21,'(''  kx            ky            kz'')')

! calculate subtractor to allow orbit to be shifted as close to the RUC origin as possible
    cp1=maxlreciplat*((((4.0D0*outlineavgxext)-1.0D0)*(p*p*u + c))-(((4.0D0*outlineavgyext) &
&-1.0D0)*(p*q*u))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(q*s)))
    cp2=maxlreciplat*((((4.0D0*outlineavgxext)-1.0D0)*(-p*q*u))+(((4.0D0*outlineavgyext) &
&-1.0D0)*(q*q*u + c))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(p*s)))
    cp3=maxlreciplat*((((4.0D0*outlineavgxext)-1.0D0)*(-q*s))-(((4.0D0*outlineavgyext) &
&-1.0D0)*(p*s))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(c)))

    cfintpointx=((ai*cp1)-(aii*cp2)+(aiii*cp3))/bigd
    cfintpointy=((bi*cp1)-(bii*cp2)+(biii*cp3))/bigd
    cfintpointz=((ci*cp1)-(cii*cp2)+(ciii*cp3))/bigd

    csubtractorx=dnint(cfintpointx)
    csubtractory=dnint(cfintpointy)
    csubtractorz=dnint(cfintpointz)

    do j=1,noutlinepts
     cp1=maxlreciplat*((((4.0D0*fsoutlinexext(j)/xlength)-1.0D0)*(p*p*u + c))-(((4.0D0*fsoutlineyext(j)/ylength) &
&-1.0D0)*(p*q*u))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(q*s)))
     cp2=maxlreciplat*((((4.0D0*fsoutlinexext(j)/xlength)-1.0D0)*(-p*q*u))+(((4.0D0*fsoutlineyext(j)/ylength) &
&-1.0D0)*(q*q*u + c))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(p*s)))
     cp3=maxlreciplat*((((4.0D0*fsoutlinexext(j)/xlength)-1.0D0)*(-q*s))-(((4.0D0*fsoutlineyext(j)/ylength) &
&-1.0D0)*(p*s))+(((4.0D0*(dble(sr2fromslicearray(i))-1.0D0)/(dble(numslices)-1.0D0))-1.0D0)*(c)))

     cfintpointx=((ai*cp1)-(aii*cp2)+(aiii*cp3))/bigd
     cfintpointy=((bi*cp1)-(bii*cp2)+(biii*cp3))/bigd
     cfintpointz=((ci*cp1)-(cii*cp2)+(ciii*cp3))/bigd

     cp1=(plrx1*(cfintpointx-csubtractorx))+(plry1*(cfintpointy-csubtractory))+(plrz1*(cfintpointz-csubtractorz))
     cp2=(plrx2*(cfintpointx-csubtractorx))+(plry2*(cfintpointy-csubtractory))+(plrz2*(cfintpointz-csubtractorz))
     cp3=(plrx3*(cfintpointx-csubtractorx))+(plry3*(cfintpointy-csubtractory))+(plrz3*(cfintpointz-csubtractorz))

     write(20,'(1x,ES13.6,1x,ES13.6,1x,ES13.6)')cp1,cp2,cp3
     write(21,'(1x,ES13.6,1x,ES13.6,1x,ES13.6)')(cp1*convau2ang),(cp2*convau2ang),(cp3*convau2ang)
    end do

    percentdone=percentdone+(1.0D0/dble(numsorted))*100.0D0
    write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,F5.1,'' %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8),percentdone

   end do

   write(*,FMT='(A1,A1,A1,A1,A1,A1,A1,''100.0 %''$)')char(8),char(8),&
&char(8),char(8),char(8),char(8),char(8)
   write(*,'('' '')')

  else

   write(20,'(''No extremal orbits found at this angle.'')')

  end if

 end if
!---end orbit outline output section

 write(*,'('' '')')
 write(*,'('' Predicted dHvA frequencies: '')')
 write(*,'('' '')')

 write(18,'('' '')')
 write(18,'('' Predicted dHvA frequencies: '')')
 write(18,'('' '')')

 write(17,'('' '')')
 write(17,'('' Predicted dHvA frequencies: '')')
 write(17,'('' '')')

 do i=1,numsorted
  write(*,'('' Freq. = '',F8.4,''+/-'',F8.4,'' kT, m* = '',F8.4,''+/-'',F8.4 &
&,'' m_e, Curv. = '',ES12.4,''+/-'',ES12.4,'' kT A^2, orbit (1=e,-1=h): '',F4.1,''+/-'',F4.1)')&
&sr2freqarray(i),sr2stdfreqarray(i),sr2massarray(i),sr2stdmassarray(i), &
&sr2curvarray(i),sr2stdcurvarray(i),sr2orbtypearray(i),sr2stdorbtypearray(i)
  write(*,'('' Orbit copies found: '',I5,'', RUC avg coords: ('',F8.4 &
&,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'')'')') &
&sr2numorbsarray(i),sr2avgxrucarray(i),sr2stdavgxrucarray(i), &
&sr2avgyrucarray(i),sr2stdavgyrucarray(i),sr2avgzrucarray(i),sr2stdavgzrucarray(i)
  write(*,'('' '')')
  write(18,'('' Freq. = '',F8.4,''+/-'',F8.4,'' kT, m* = '',F8.4,''+/-'',F8.4 &
&,'' m_e, Curv. = '',ES12.4,''+/-'',ES12.4,'' kT A^2, orbit (1=e,-1=h): '',F4.1,''+/-'',F4.1)')&
&sr2freqarray(i),sr2stdfreqarray(i),sr2massarray(i),sr2stdmassarray(i), &
&sr2curvarray(i),sr2stdcurvarray(i),sr2orbtypearray(i),sr2stdorbtypearray(i)
  write(18,'('' Orbit copies found: '',I5,'', RUC avg coords: ('',F8.4 &
&,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'')'')') &
&sr2numorbsarray(i),sr2avgxrucarray(i),sr2stdavgxrucarray(i), &
&sr2avgyrucarray(i),sr2stdavgyrucarray(i),sr2avgzrucarray(i),sr2stdavgzrucarray(i)
  write(18,'('' '')')
  write(19,'(F10.6,'','',F10.6,'','',ES14.6,'','',ES14.6,'','',ES14.6,'','',F6.3,'','',I5)')&
&180.0D0*theta/pi,180.0D0*phi/pi,sr2freqarray(i),sr2massarray(i),sr2curvarray(i),&
&sr2orbtypearray(i),sr2numorbsarray(i)
  write(17,'('' Freq. = '',F8.4,''+/-'',F8.4,'' kT, m* = '',F8.4,''+/-'',F8.4 &
&,'' m_e, Curv. = '',ES12.4,''+/-'',ES12.4,'' kT A^2, orbit (1=e,-1=h): '',F4.1,''+/-'',F4.1)')&
&sr2freqarray(i),sr2stdfreqarray(i),sr2massarray(i),sr2stdmassarray(i), &
&sr2curvarray(i),sr2stdcurvarray(i),sr2orbtypearray(i),sr2stdorbtypearray(i)
  write(17,'('' Orbit copies found: '',I5,'', RUC avg coords: ('',F8.4 &
&,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'','',F8.4,''+/-'',F8.4,'')'')') &
&sr2numorbsarray(i),sr2avgxrucarray(i),sr2stdavgxrucarray(i), &
&sr2avgyrucarray(i),sr2stdavgyrucarray(i),sr2avgzrucarray(i),sr2stdavgzrucarray(i)
  if (hvd/='r') then
   write(17,'('' Orbit copy chosen for outline: orbit '',I5,'' from slice '',I4)')sr2orbnumarray(i),sr2fromslicearray(i)
  end if
  write(17,'('' '')')
 end do

 percentelectronlike = 100.0D0*bandelectronarea/(bandelectronarea+bandholearea)
 percentholelike = 100.0D0*bandholearea/(bandelectronarea+bandholearea)
 if (hvd/='r') then
  write(*,'('' Comparing the volumes enclosed by electron and hole orbits (at this angle),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
  write(*,'('' '')')
  write(18,'('' Comparing the volumes enclosed by electron and hole orbits (at this angle),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
  write(18,'('' '')')
 end if

 write(17,'('' Comparing the volumes enclosed by electron and hole orbits (at this angle),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
 write(17,'('' '')')

 rotelectronarea=rotelectronarea+bandelectronarea
 rotholearea=rotholearea+bandholearea

 write(17,'('' Array bounds (read/defined): Max numslfs(slice) = '',I6,'' / '',I6, &
&'', numchunks = '',I6,'' / '',I6,'', numnd = '',I6,'' / '',I6 &
&)')MAXVAL(numslfs),measpardim,numchunks,measpardim2,numnd,measpardim2
 write(17,'('' '')')

 if (MAXVAL(numslfs)>measpardim) then
  write(*,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(*,'('' Max numslfs(slice) = '',I6,'' measpardim = '',I6)')MAXVAL(numslfs),measpardim
  write(*,'('' '')')
  write(17,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(17,'('' Max numslfs(slice) = '',I6,'' measpardim = '',I6)')MAXVAL(numslfs),measpardim
  write(17,'('' '')')
  write(18,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(18,'('' Max numslfs(slice) = '',I6,'' measpardim = '',I6)')MAXVAL(numslfs),measpardim
  write(18,'('' '')')
 end if

 if (numchunks>measpardim2) then
  write(*,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(*,'('' numchunks = '',I6,'' measpardim2 = '',I6)')numchunks,measpardim2
  write(*,'('' '')')
  write(17,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(17,'('' numchunks = '',I6,'' measpardim2 = '',I6)')numchunks,measpardim2
  write(17,'('' '')')
  write(18,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(18,'('' numchunks = '',I6,'' measpardim2 = '',I6)')numchunks,measpardim2
  write(18,'('' '')')
 end if

 if (numnd>measpardim2) then
  write(*,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(*,'('' numnd = '',I6,'' measpardim2 = '',I6)')numnd,measpardim2
  write(*,'('' '')')
  write(17,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(17,'('' numnd = '',I6,'' measpardim2 = '',I6)')numnd,measpardim2
  write(17,'('' '')')
  write(18,'('' WARNING: Read beyond array bounds! Do not trust these results!'')')
  write(18,'('' numnd = '',I6,'' measpardim2 = '',I6)')numnd,measpardim2
  write(18,'('' '')')
 end if

 if (rotstepper==1) then
  CALL ptimediff(rottimehr,rottimemin,rottimesec,rottimems)
  CALL ptimediff(matchtimehr,matchtimemin,matchtimesec,matchtimems)
 end if

end do

if (hvd=='r') then
 percentelectronlike = 100.0D0*rotelectronarea/(rotelectronarea+rotholearea)
 percentholelike = 100.0D0*rotholearea/(rotelectronarea+rotholearea)
 write(*,'('' '')')
 write(*,'('' Comparing the volumes enclosed by electron and hole orbits (angle-averaged),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
 write(*,'('' '')')
 write(17,'('' '')')
 write(17,'('' Comparing the volumes enclosed by electron and hole orbits (angle-averaged),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
 write(17,'('' '')')
 write(18,'('' '')')
 write(18,'('' Comparing the volumes enclosed by electron and hole orbits (angle-averaged),'' &
&,'' this band is estimated to be '',F5.1,''% electron-like and '',F5.1 &
&,''% hole-like.'')')percentelectronlike,percentholelike
 write(18,'('' '')')
end if

write(*,'('' '')')
write(*,'('' All done! '')')
write(*,'('' '')')
write(*,'('' '')')

end if

CALL ptimediff(tottimehr,tottimemin,tottimesec,tottimems)

write(17,'('' The interpolation and orbit detection for the first angle took '',I2,''hr:'', &
&I2,''min:'',I2,''.'',I3.3,''s'')')dettimehr,dettimemin,dettimesec,dettimems
write(17,'('' '')')
write(18,'('' The interpolation and orbit detection for the first angle took '',I2,''hr:'', &
&I2,''min:'',I2,''.'',I3.3,''s'')')dettimehr,dettimemin,dettimesec,dettimems
write(18,'('' '')')

write(17,'('' The FS matching/extremum finding for the first angle took '', &
&I2,''hr:'',I2,''min:'',I2,''.'',I3.3,''s'')')matchtimehr,matchtimemin,matchtimesec,matchtimems
write(17,'('' '')')
write(18,'('' The FS matching/extremum finding for the first angle took '', &
&I2,''hr:'',I2,''min:'',I2,''.'',I3.3,''s'')')matchtimehr,matchtimemin,matchtimesec,matchtimems
write(18,'('' '')')

write(17,'('' Calculations for one angle took '',I2,''hr:'',I2,''min:'', &
&I2,''.'',I3.3,''s'')')rottimehr,rottimemin,rottimesec,rottimems
write(17,'('' '')')
write(18,'('' Calculations for one angle took '',I2,''hr:'',I2,''min:'', &
&I2,''.'',I3.3,''s'')')rottimehr,rottimemin,rottimesec,rottimems
write(18,'('' '')')

write(17,'('' Whole program run took '',I2,''hr:'',I2,''min:'',I2,''.'', &
&I3.3,''s'')')tottimehr,tottimemin,tottimesec,tottimems
write(17,'('' '')')
write(18,'('' Whole program run took '',I2,''hr:'',I2,''min:'',I2,''.'', &
&I3.3,''s'')')tottimehr,tottimemin,tottimesec,tottimems
write(18,'('' '')')

CALL DATE_AND_TIME(date=pdate,time=ptime,zone=ptimezone,values=ptimedatevalues)
write(17,'('' Program finished on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'', &
&I2,'':'',I2,''.'',I3.3)')ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1), &
ptimedatevalues(5),ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)
write(18,'('' Program finished on '',I2,''/'',I2,''/'',I4,'' at '',I2,'':'', &
&I2,'':'',I2,''.'',I3.3)')ptimedatevalues(3),ptimedatevalues(2),ptimedatevalues(1), &
ptimedatevalues(5),ptimedatevalues(6),ptimedatevalues(7),ptimedatevalues(8)

endfile 17
close(17,status='keep')

endfile 18
close(18,status='keep')

endfile 19
close(19,status='keep')

endfile 20
close(20,status='keep')

endfile 21
close(20,status='keep')

STOP

CONTAINS

! ******************************************************************************

 SUBROUTINE preadbxsf
  open(15,file=filename,status='old')

  keepreading=.TRUE.
  do while (keepreading.eqv..TRUE.)
   read(15,fmt='(A100)')tempstringread
   tempstringread=TRIM(tempstringread)
   if ((INDEX(tempstringread,'Fermi Energy')/=0).OR.(INDEX(tempstringread,'fermi energy')/=0)&
&.OR.(INDEX(tempstringread,'FERMI ENERGY')/=0).OR.(INDEX(tempstringread,'Fermi energy')/=0)) then
    tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
    read(tempstringread,*)readfermienergy
    keepreading=.FALSE.
   end if
  end do
  keepreading=.TRUE.

  do i=1,4
   read(15,*)tempstringread
  end do

  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread,*)numlistedbands

  if (numlistedbands/=1.0D0) then
   write(*,'('' '')')
   write(*,'('' ***BXSF ERROR: Header indicates that file contains multiple bands!***'')')
   write(*,'('' (Need to separate each band into its own BXSF file, usually by'')')
   write(*,'(''   opening in XCrysDen and re-saving individual bands from there.)'')')
   write(*,'('' '')')
   STOP
  end if

  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)nx
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)ny
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'0123456789'):)
  read(tempstringread,*)nz

  if ((nx>maxnuminput).OR.(ny>maxnuminput).OR.(nz>maxnuminput)) then
   write(*,'('' '')')
   write(*,'('' ***BXSF ERROR: too many k-points for current array bounds!***'')')
   write(*,'('' (Please increase the value of the maxnuminput parameter in the'')')
   write(*,'(''  source code for this program and try compiling and running again.)'')')
   write(*,'('' '')')
   STOP
  end if

  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplato1
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplato2
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread,*)patricklreciplato3

  if ((patricklreciplato1/=0.0D0).OR.(patricklreciplato2/=0.0D0).OR.(patricklreciplato3/=0.0D0)) then
   write(*,'('' '')')
   write(*,'('' ***BXSF ERROR: reciprocal lattice vector origin is not at (0, 0, 0)!***'')')
   write(*,'('' '')')
   STOP
  end if

  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplatx1
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplatx2
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread,*)patricklreciplatx3

  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplaty1
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplaty2
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread,*)patricklreciplaty3
 
  read(15,fmt='(A100)')tempstringread
  tempstringread=TRIM(tempstringread)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplatz1
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread(:SCAN(tempstringread,' ')-1),*)patricklreciplatz2
  tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
  tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
  read(tempstringread,*)patricklreciplatz3

  read(15,*)tempstringread

  numkpoints=nx*ny*nz
  numread=0
  keepreading=.TRUE.
  do while (keepreading.eqv..TRUE.)
   read(15,fmt='(A100)')tempstringread
   tempstringread=TRIM(tempstringread)
   if ((INDEX(tempstringread,'BAND:')/=0).OR.(INDEX(tempstringread,'Band:')/=0)&
&.OR.(INDEX(tempstringread,'band:')/=0).OR.(INDEX(tempstringread,'prod:')/=0)&
&.OR.(INDEX(tempstringread,'Prod:')/=0).OR.(INDEX(tempstringread,'PROD:')/=0)) then
    write(*,'('' '')')
    write(*,'('' ***BXSF ERROR: File contains multiple bands!***'')')
    write(*,'('' (Need to separate each band into its own BXSF file, usually by'')')
    write(*,'(''   opening in XCrysDen and re-saving individual bands from there.)'')')
    write(*,'('' '')')
    STOP
   end if
   if ((INDEX(tempstringread,'END')==0).AND.(INDEX(tempstringread,'End')==0)&
&.AND.(INDEX(tempstringread,'end')==0)) then
    tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
    do while (SCAN(tempstringread,'-0123456789')/=0)
     numread = numread + 1
     read(tempstringread(:SCAN(tempstringread,' ')-1),*)kreadarray(numread)
     tempstringread=tempstringread(SCAN(tempstringread,' ')+1:)
     tempstringread=tempstringread(SCAN(tempstringread,'-0123456789'):)
    end do
   else
    keepreading=.FALSE.
   end if
  end do

  do i=1,nx
   do j=1,ny
    do k=1,nz
     masterkarray(i,j,k)=kreadarray(k+(j-1)*nz+(i-1)*nz*ny)
    end do
   end do
  end do
  close(15,status='keep')

  if (masterkarray(nx,ny,nz)/=masterkarray(1,1,1)) then
   write(*,'('' ***BXSF ERROR: Energies appear to be specified on a Periodic Grid rather than a General Grid!***'')')
   write(*,'('' (Are you using the ELK or exciting DFT packages? Please see the'')')
   write(*,'(''   README-forSKEAF.txt file for notes and further instructions.)'')')
   write(*,'('' '')')
   STOP
  end if

  patricklreciplatx=dsqrt(dabs((patricklreciplatx1*patricklreciplatx1)+ &
&(patricklreciplatx2*patricklreciplatx2)+ & 
&(patricklreciplatx3*patricklreciplatx3)))
  patricklreciplaty=dsqrt(dabs((patricklreciplaty1*patricklreciplaty1)+ &
&(patricklreciplaty2*patricklreciplaty2)+ & 
&(patricklreciplaty3*patricklreciplaty3)))
  patricklreciplatz=dsqrt(dabs((patricklreciplatz1*patricklreciplatz1)+ &
&(patricklreciplatz2*patricklreciplatz2)+ &
&(patricklreciplatz3*patricklreciplatz3)))

  plrx1=2.0D0*pi*patricklreciplatx1/convau2ang
  plrx2=2.0D0*pi*patricklreciplatx2/convau2ang
  plrx3=2.0D0*pi*patricklreciplatx3/convau2ang
  plry1=2.0D0*pi*patricklreciplaty1/convau2ang
  plry2=2.0D0*pi*patricklreciplaty2/convau2ang
  plry3=2.0D0*pi*patricklreciplaty3/convau2ang
  plrz1=2.0D0*pi*patricklreciplatz1/convau2ang
  plrz2=2.0D0*pi*patricklreciplatz2/convau2ang
  plrz3=2.0D0*pi*patricklreciplatz3/convau2ang

  lreciplatx=dsqrt(dabs((plrx1*plrx1)+(plrx2*plrx2)+(plrx3*plrx3)))
  lreciplaty=dsqrt(dabs((plry1*plry1)+(plry2*plry2)+(plry3*plry3)))
  lreciplatz=dsqrt(dabs((plrz1*plrz1)+(plrz2*plrz2)+(plrz3*plrz3)))
  maxlreciplat=dmax1(lreciplatx,lreciplaty,lreciplatz)

  kpointspacingx=lreciplatx/dble(nx-1)
  kpointspacingy=lreciplaty/dble(ny-1)
  kpointspacingz=lreciplatz/dble(nz-1)
  minkpointspacing=dmin1(kpointspacingx,kpointspacingy,kpointspacingz)
  minnumint=ceiling(maxlreciplat/minkpointspacing)+1

  anglelatxy=dacos(((plrx1*plry1)+(plrx2*plry2)+(plrx3*plry3))/(lreciplatx*lreciplaty))
  anglelatxz=dacos(((plrx1*plrz1)+(plrx2*plrz2)+(plrx3*plrz3))/(lreciplatx*lreciplatz))
  anglelatyz=dacos(((plry1*plrz1)+(plry2*plrz2)+(plry3*plrz3))/(lreciplaty*lreciplatz))

  freq2kxy=2.0D0*convfsarea2kt*kpointspacingx*kpointspacingy*dsin(anglelatxy)
  freq2kxz=2.0D0*convfsarea2kt*kpointspacingx*kpointspacingz*dsin(anglelatxz)
  freq2kyz=2.0D0*convfsarea2kt*kpointspacingy*kpointspacingz*dsin(anglelatyz)
  freq2kmax=dmax1(freq2kxy,freq2kxz,freq2kyz)

 END SUBROUTINE preadbxsf

! ******************************************************************************

 SUBROUTINE psetangle(tmphvd,tmptheta,tmpphi)
  CHARACTER, INTENT(IN) :: tmphvd*100
  DOUBLE PRECISION, INTENT(OUT) :: tmptheta,tmpphi
   if (tmphvd=='c') then
    if (plrz1==0.0D0.AND.plrz2==0.0D0) then
     tmptheta=0.0D0
     tmpphi=0.0D0
    else
     tmptheta=datan2(plrz2,plrz1)
     tmpphi=dacos(plrz3/dsqrt(((plrz1)*(plrz1))+((plrz2)*(plrz2))+((plrz3)*(plrz3))))
    end if
   end if
   if (tmphvd=='b') then
    if (plry1==0.0D0.AND.plry2==0.0D0) then
     tmptheta=0.0D0
     tmpphi=0.0D0
    else
     tmptheta=datan2(plry2,plry1)
     tmpphi=dacos(plry3/dsqrt(((plry1)*(plry1))+((plry2)*(plry2))+((plry3)*(plry3))))
    end if
   end if
   if (tmphvd=='a') then
    if (plrx1==0.0D0.AND.plrx2==0.0D0) then
     tmptheta=0.0D0
     tmpphi=0.0D0
    else
     tmptheta=datan2(plrx2,plrx1)
     tmpphi=dacos(plrx3/dsqrt(((plrx1)*(plrx1))+((plrx2)*(plrx2))+((plrx3)*(plrx3))))
    end if
   end if
   if (tmphvd/='a'.AND.tmphvd/='b'.AND.tmphvd/='c'.AND.tmphvd/='n'.AND.tmphvd/='r') then
    write(*,'('' Bad input '')')
    badinput=.TRUE.
   end if 

   ! non-princ. H-vector
   if (tmphvd=='n') then
    write(*,'('' '')')
    write(*,'(''  Theta (angle from x-axis toward y-axis) in degrees? [0-360] '')')
    read(*,*)tmptheta
    if (tmptheta<0.0D0) then
     tmptheta=0.0D0
    end if
    if (tmptheta>=360.0D0) then
     tmptheta=0.0D0
    end if
    write(*,'('' '')')
    write(*,'(''  Phi (angle down from z-axis) in degrees? [0-180] '')')
    read(*,*)tmpphi
    if (tmpphi<0.0D0) then
     tmpphi=0.0D0
    end if
    if (tmpphi>180.0D0) then
     tmpphi=180.0D0
    end if
    tmptheta=tmptheta*pi/180.0D0
    tmpphi=tmpphi*pi/180.0D0
  end if 
 END SUBROUTINE psetangle

! ******************************************************************************

 SUBROUTINE slicedos(tk,tmpdoscontrib,tmpocc,tmpempty)
  INTEGER(KIND=int8byte), INTENT(IN) :: tk
  DOUBLE PRECISION, INTENT(OUT) :: tmpdoscontrib
  INTEGER(KIND=int8byte), INTENT(OUT) :: tmpocc, tmpempty

  integer(KIND=int8byte) :: ti
  integer(KIND=int8byte) :: tj
  double precision::dintpointx
  double precision::dintpointy
  double precision::dintpointz
  double precision::dintpointzplus1
  double precision,allocatable,dimension(:,:)::kslice
  double precision,allocatable,dimension(:,:)::ksliceplus1
  double precision,dimension(8)::microcelle
  double precision,dimension(4)::tetraunsorte
  double precision,dimension(4)::tetrae
  integer(KIND=int8byte)::tetit
  double precision,dimension(2)::tsortingtemp
  integer(KIND=int8byte),dimension(1)::tetramaxloc
  integer(KIND=int8byte),dimension(1)::tetraminloc

 allocate(kslice(scarraydimension,scarraydimension),ksliceplus1(scarraydimension,scarraydimension))

 tmpdoscontrib = 0.0D0
 tmpocc = 0
 tmpempty = 0

 dintpointz=(((dble(tk)-1.0D0)/(dble(4*numint)-1.0D0))*(dble(nz)-1.0D0))+1.0D0
 dintpointzplus1=(((dble(tk+1)-1.0D0)/(dble(4*numint)-1.0D0))*(dble(nz)-1.0D0))+1.0D0
 do tj=1,4*numint
  dintpointy=(((dble(tj)-1.0D0)/(dble(4*numint)-1.0D0))*(dble(ny)-1.0D0))+1.0D0
  do ti=1,4*numint
   dintpointx=(((dble(ti)-1.0D0)/(dble(4*numint)-1.0D0))*(dble(nx)-1.0D0))+1.0D0
   kslice(ti,tj)=pinterpolation(dintpointx,dintpointy,dintpointz)
   ksliceplus1(ti,tj)=pinterpolation(dintpointx,dintpointy,dintpointzplus1)
  end do
 end do

 do tj=1,((4*numint)-1)
  do ti=1,((4*numint)-1)

   if (kslice(ti,tj)<=fermienergy) then
    tmpocc=tmpocc+1
   else
    tmpempty=tmpempty+1
   end if

   microcelle(1)=kslice(ti,tj)
   microcelle(2)=kslice(ti+1,tj)
   microcelle(3)=kslice(ti,tj+1)
   microcelle(4)=kslice(ti+1,tj+1)
   microcelle(5)=ksliceplus1(ti,tj)
   microcelle(6)=ksliceplus1(ti+1,tj)
   microcelle(7)=ksliceplus1(ti,tj+1)
   microcelle(8)=ksliceplus1(ti+1,tj+1)
   if ((fermienergy>=MINVAL(microcelle)).AND.(fermienergy<=MAXVAL(microcelle))) then
    do tetit=1,6

     if (tetit==1) then
      tetraunsorte(1)=microcelle(1)
      tetraunsorte(2)=microcelle(2)
      tetraunsorte(3)=microcelle(3)
      tetraunsorte(4)=microcelle(6)
     end if
     if (tetit==2) then
      tetraunsorte(1)=microcelle(2)
      tetraunsorte(2)=microcelle(3)
      tetraunsorte(3)=microcelle(4)
      tetraunsorte(4)=microcelle(6)
     end if
     if (tetit==3) then
      tetraunsorte(1)=microcelle(3)
      tetraunsorte(2)=microcelle(4)
      tetraunsorte(3)=microcelle(6)
      tetraunsorte(4)=microcelle(8)
     end if
     if (tetit==4) then
      tetraunsorte(1)=microcelle(3)
      tetraunsorte(2)=microcelle(6)
      tetraunsorte(3)=microcelle(7)
      tetraunsorte(4)=microcelle(8)
     end if
     if (tetit==5) then
      tetraunsorte(1)=microcelle(3)
      tetraunsorte(2)=microcelle(5)
      tetraunsorte(3)=microcelle(6)
      tetraunsorte(4)=microcelle(7)
     end if
     if (tetit==6) then
      tetraunsorte(1)=microcelle(1)
      tetraunsorte(2)=microcelle(3)
      tetraunsorte(3)=microcelle(5)
      tetraunsorte(4)=microcelle(6)
     end if

     if ((fermienergy>=MINVAL(tetraunsorte)).AND.(fermienergy<=MAXVAL(tetraunsorte))) then

! sort the energies from lowest to highest
      tetrae(1)=MINVAL(tetraunsorte)
      tetrae(4)=MAXVAL(tetraunsorte)
      tetramaxloc=MAXLOC(tetraunsorte)
      tetraminloc=MINLOC(tetraunsorte)
      if ((tetramaxloc(1)==1).OR.(tetraminloc(1)==1)) then
       if ((tetramaxloc(1)==2).OR.(tetraminloc(1)==2)) then
        tsortingtemp(1)=tetraunsorte(3)
        tsortingtemp(2)=tetraunsorte(4)
       else
        tsortingtemp(1)=tetraunsorte(2)
        if ((tetramaxloc(1)==3).OR.(tetraminloc(1)==3)) then
         tsortingtemp(2)=tetraunsorte(4)
        else
         tsortingtemp(2)=tetraunsorte(3)
        end if
       end if
      else
       tsortingtemp(1)=tetraunsorte(1)
       if ((tetramaxloc(1)/=2).AND.(tetraminloc(1)/=2)) then
        tsortingtemp(2)=tetraunsorte(2)
       elseif ((tetramaxloc(1)/=3).AND.(tetraminloc(1)/=3)) then
        tsortingtemp(2)=tetraunsorte(3)
       else
        tsortingtemp(2)=tetraunsorte(4)
       end if
      end if
      tetrae(2)=MINVAL(tsortingtemp)
      tetrae(3)=MAXVAL(tsortingtemp)

! calculate tetrahedron DOS contribution
      if ((fermienergy>=tetrae(1)).AND.(fermienergy<=tetrae(2))) then
       if (tetrae(1)/=tetrae(2)) then
       tmpdoscontrib=tmpdoscontrib+((3.0D0*tetrahedronvolume &
&*(fermienergy-tetrae(1))*(fermienergy-tetrae(1)))/((tetrae(2)-tetrae(1)) &
&*(tetrae(3)-tetrae(1))*(tetrae(4)-tetrae(1))))
       end if

      elseif ((fermienergy>=tetrae(3)).AND.(fermienergy<=tetrae(4))) then
       if (tetrae(3)/=tetrae(4)) then
      tmpdoscontrib=tmpdoscontrib+((3.0D0*tetrahedronvolume &
&*(tetrae(4)-fermienergy)*(tetrae(4)-fermienergy))/((tetrae(4)-tetrae(1)) &
&*(tetrae(4)-tetrae(2))*(tetrae(4)-tetrae(3))))
       end if

      else
       tmpdoscontrib=tmpdoscontrib+(((3.0D0*tetrahedronvolume) &
&/((tetrae(3)-tetrae(1))*(tetrae(4)-tetrae(1))))*(tetrae(2)-tetrae(1) &
&+(2.0D0*fermienergy)-(2.0D0*tetrae(2))-(((tetrae(3)-tetrae(1)+tetrae(4) &
&-tetrae(2))*(fermienergy-tetrae(2))*(fermienergy-tetrae(2)))/ &
&((tetrae(3)-tetrae(2))*(tetrae(4)-tetrae(2))))))

      end if

     end if

    end do
   end if
  end do
 end do

 END SUBROUTINE slicedos

! ******************************************************************************

 DOUBLE PRECISION FUNCTION pinterpolation(inputx,inputy,inputz)
  DOUBLE PRECISION, INTENT(IN) :: inputx, inputy, inputz

  double precision::intpointx
  double precision::intpointy
  double precision::intpointz
  integer(KIND=int8byte)::nintpointx
  integer(KIND=int8byte)::nintpointy
  integer(KIND=int8byte)::nintpointz
  integer(KIND=int8byte)::x1
  integer(KIND=int8byte)::x2
  integer(KIND=int8byte)::x3
  integer(KIND=int8byte)::x4
  integer(KIND=int8byte)::x1red
  integer(KIND=int8byte)::x2red
  integer(KIND=int8byte)::x3red
  integer(KIND=int8byte)::x4red
  integer(KIND=int8byte)::y1
  integer(KIND=int8byte)::y2
  integer(KIND=int8byte)::y3
  integer(KIND=int8byte)::y4
  integer(KIND=int8byte)::y1red
  integer(KIND=int8byte)::y2red
  integer(KIND=int8byte)::y3red
  integer(KIND=int8byte)::y4red
  integer(KIND=int8byte)::z1
  integer(KIND=int8byte)::z2
  integer(KIND=int8byte)::z3
  integer(KIND=int8byte)::z4
  integer(KIND=int8byte)::z1red
  integer(KIND=int8byte)::z2red
  integer(KIND=int8byte)::z3red
  integer(KIND=int8byte)::z4red
  double precision::ep1
  double precision::ep2
  double precision::ep3
  double precision::ep4
  double precision::ep11
  double precision::ep12
  double precision::ep13
  double precision::ep14
  double precision::ep21
  double precision::ep22
  double precision::ep23
  double precision::ep24
  double precision::ep31
  double precision::ep32
  double precision::ep33
  double precision::ep34
  double precision::ep41
  double precision::ep42
  double precision::ep43
  double precision::ep44

    intpointx=inputx
    intpointy=inputy
    intpointz=inputz

    nintpointx=idnint(intpointx)
    nintpointy=idnint(intpointy)
    nintpointz=idnint(intpointz)

    if (nintpointz==intpointz) then
     if (nintpointy==intpointy) then
      if (nintpointx==intpointx) then
       ! intpointx, y and z are integers
       pinterpolation=masterkarray(nintpointx,nintpointy,nintpointz)
      else
       !intpointy and z are integers, but x is not; 1D
       x2=floor(intpointx)
       x3=ceiling(intpointx)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=nx-1
       end if
       if (x3==nx) then
        x4red=2
       end if
       pinterpolation=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/ &
&((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,nintpointy,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,nintpointy,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,nintpointy,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,nintpointy,nintpointz))

      end if
     else
      ! intpointz is integer but intpointy is not
      if (nintpointx==intpointx) then
       ! intpointz and x are integers but intpointy is not; 1D
       x2=floor(intpointy)
       x3=ceiling(intpointy)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=ny-1
       end if
       if (x3==ny) then
        x4red=2
       end if
       pinterpolation=((((intpointy-x2)*(intpointy-x3)*(intpointy-x4))/ &
&((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,x1red,nintpointz))+ & 
&        ((((intpointy-x1)*(intpointy-x3)*(intpointy-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,x2red,nintpointz))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,x3red,nintpointz))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,x4red,nintpointz))

      else
       ! intpointx and y are not integers, but z is; 2D
       x2=floor(intpointx)
       x3=ceiling(intpointx)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=nx-1
       end if
       if (x3==nx) then
        x4red=2
       end if
       y2=floor(intpointy)
       y3=ceiling(intpointy)
       y1=y2-1
       y4=y3+1
       y1red=y1
       y2red=y2
       y3red=y3
       y4red=y4
       if (y2==1) then
        y1red=ny-1
       end if
       if (y3==ny) then
        y4red=2
       end if

       ep1=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,y1red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,y1red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,y1red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,y1red,nintpointz))

       ep2=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,y2red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,y2red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,y2red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,y2red,nintpointz))

       ep3=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,y3red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,y3red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,y3red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,y3red,nintpointz))

       ep4=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,y4red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,y4red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,y4red,nintpointz))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,y4red,nintpointz))

       pinterpolation=((((intpointy-y2)*(intpointy-y3)*(intpointy-y4))/((y1-y2)*(y1-y3)*(y1-y4)))*ep1)+ & 
&        ((((intpointy-y1)*(intpointy-y3)*(intpointy-y4))/((y2-y1)*(y2-y3)*(y2-y4)))*ep2)+ & 
&        ((((intpointy-y1)*(intpointy-y2)*(intpointy-y4))/((y3-y1)*(y3-y2)*(y3-y4)))*ep3)+ & 
&        ((((intpointy-y1)*(intpointy-y2)*(intpointy-y3))/((y4-y1)*(y4-y2)*(y4-y3)))*ep4)

      end if

    end if
    else
     ! intpointz is not an integer
     if (nintpointx==intpointx) then
      ! intpointx is an integer, z is not
      if (nintpointy==intpointy) then
       ! intpointx and y are integers, but z is not; 1D
       x2=floor(intpointz)
       x3=ceiling(intpointz)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=nz-1
       end if
       if (x3==nz) then
        x4red=2
       end if
       pinterpolation=((((intpointz-x2)*(intpointz-x3)*(intpointz-x4))/ &
&((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,nintpointy,x1red))+ & 
&        ((((intpointz-x1)*(intpointz-x3)*(intpointz-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,nintpointy,x2red))+ & 
&        ((((intpointz-x1)*(intpointz-x2)*(intpointz-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,nintpointy,x3red))+ & 
&        ((((intpointz-x1)*(intpointz-x2)*(intpointz-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,nintpointy,x4red))

      else
       ! intpointx is an integer, y and z are not; 2D
       x2=floor(intpointy)
       x3=ceiling(intpointy)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=ny-1
       end if
       if (x3==ny) then
        x4red=2
       end if
       y2=floor(intpointz)
       y3=ceiling(intpointz)
       y1=y2-1
       y4=y3+1
       y1red=y1
       y2red=y2
       y3red=y3
       y4red=y4
       if (y2==1) then
        y1red=nz-1
       end if
       if (y3==nz) then
        y4red=2
       end if

       ep1=((((intpointy-x2)*(intpointy-x3)*(intpointy-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,x1red,y1red))+ & 
&        ((((intpointy-x1)*(intpointy-x3)*(intpointy-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,x2red,y1red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,x3red,y1red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,x4red,y1red))

       ep2=((((intpointy-x2)*(intpointy-x3)*(intpointy-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,x1red,y2red))+ & 
&        ((((intpointy-x1)*(intpointy-x3)*(intpointy-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,x2red,y2red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,x3red,y2red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,x4red,y2red))

       ep3=((((intpointy-x2)*(intpointy-x3)*(intpointy-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,x1red,y3red))+ & 
&        ((((intpointy-x1)*(intpointy-x3)*(intpointy-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,x2red,y3red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,x3red,y3red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,x4red,y3red))

       ep4=((((intpointy-x2)*(intpointy-x3)*(intpointy-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(nintpointx,x1red,y4red))+ & 
&        ((((intpointy-x1)*(intpointy-x3)*(intpointy-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(nintpointx,x2red,y4red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(nintpointx,x3red,y4red))+ & 
&        ((((intpointy-x1)*(intpointy-x2)*(intpointy-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(nintpointx,x4red,y4red))

       pinterpolation=((((intpointz-y2)*(intpointz-y3)*(intpointz-y4))/((y1-y2)*(y1-y3)*(y1-y4)))*ep1)+ & 
&        ((((intpointz-y1)*(intpointz-y3)*(intpointz-y4))/((y2-y1)*(y2-y3)*(y2-y4)))*ep2)+ & 
&        ((((intpointz-y1)*(intpointz-y2)*(intpointz-y4))/((y3-y1)*(y3-y2)*(y3-y4)))*ep3)+ & 
&        ((((intpointz-y1)*(intpointz-y2)*(intpointz-y3))/((y4-y1)*(y4-y2)*(y4-y3)))*ep4)
      end if
     else
      ! intpointx and z are not integers

      if (nintpointy==intpointy) then
       ! intpointy is an integer, x and z are not; 2D
       x2=floor(intpointx)
       x3=ceiling(intpointx)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=nx-1
       end if
       if (x3==nx) then
        x4red=2
       end if
       y2=floor(intpointz)
       y3=ceiling(intpointz)
       y1=y2-1
       y4=y3+1
       y1red=y1
       y2red=y2
       y3red=y3
       y4red=y4
       if (y2==1) then
        y1red=nz-1
       end if
       if (y3==nz) then
        y4red=2
       end if

       ep1=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,nintpointy,y1red))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,nintpointy,y1red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,nintpointy,y1red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,nintpointy,y1red))

       ep2=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,nintpointy,y2red))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,nintpointy,y2red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,nintpointy,y2red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,nintpointy,y2red))

       ep3=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,nintpointy,y3red))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,nintpointy,y3red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,nintpointy,y3red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,nintpointy,y3red))

       ep4=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*masterkarray(x1red,nintpointy,y4red))+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*masterkarray(x2red,nintpointy,y4red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*masterkarray(x3red,nintpointy,y4red))+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*masterkarray(x4red,nintpointy,y4red))

       pinterpolation=((((intpointz-y2)*(intpointz-y3)*(intpointz-y4))/((y1-y2)*(y1-y3)*(y1-y4)))*ep1)+ & 
&        ((((intpointz-y1)*(intpointz-y3)*(intpointz-y4))/((y2-y1)*(y2-y3)*(y2-y4)))*ep2)+ & 
&        ((((intpointz-y1)*(intpointz-y2)*(intpointz-y4))/((y3-y1)*(y3-y2)*(y3-y4)))*ep3)+ & 
&        ((((intpointz-y1)*(intpointz-y2)*(intpointz-y3))/((y4-y1)*(y4-y2)*(y4-y3)))*ep4)
      else
       ! none of intpointx, y or z are integers; 3D

       x2=floor(intpointx)
       x3=ceiling(intpointx)
       x1=x2-1
       x4=x3+1
       x1red=x1
       x2red=x2
       x3red=x3
       x4red=x4
       if (x2==1) then
        x1red=nx-1
       end if
       if (x3==nx) then
        x4red=2
       end if
       y2=floor(intpointy)
       y3=ceiling(intpointy)
       y1=y2-1
       y4=y3+1
       y1red=y1
       y2red=y2
       y3red=y3
       y4red=y4
       if (y2==1) then
        y1red=ny-1
       end if
       if (y3==ny) then
        y4red=2
       end if
       z2=floor(intpointz)
       z3=ceiling(intpointz)
       z1=z2-1
       z4=z3+1
       z1red=z1
       z2red=z2
       z3red=z3
       z4red=z4
       if (z2==1) then
        z1red=nz-1
       end if
       if (z3==nz) then
        z4red=2
       end if

       ep11=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x1red,y1red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x1red,y1red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x1red,y1red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x1red,y1red,z4red))

       ep12=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x2red,y1red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x2red,y1red,z2red))+ & 

&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x2red,y1red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x2red,y1red,z4red))

       ep13=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x3red,y1red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x3red,y1red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x3red,y1red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x3red,y1red,z4red))

       ep14=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x4red,y1red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x4red,y1red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x4red,y1red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x4red,y1red,z4red))

       ep21=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x1red,y2red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x1red,y2red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x1red,y2red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x1red,y2red,z4red))

       ep22=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x2red,y2red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x2red,y2red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x2red,y2red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x2red,y2red,z4red))

       ep23=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x3red,y2red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x3red,y2red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x3red,y2red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x3red,y2red,z4red))

       ep24=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x4red,y2red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x4red,y2red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x4red,y2red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x4red,y2red,z4red))

       ep31=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x1red,y3red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x1red,y3red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x1red,y3red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x1red,y3red,z4red))

       ep32=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x2red,y3red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x2red,y3red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x2red,y3red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x2red,y3red,z4red))

       ep33=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x3red,y3red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x3red,y3red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x3red,y3red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x3red,y3red,z4red))

       ep34=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x4red,y3red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x4red,y3red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x4red,y3red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x4red,y3red,z4red))

       ep41=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x1red,y4red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x1red,y4red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x1red,y4red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x1red,y4red,z4red))

       ep42=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x2red,y4red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x2red,y4red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x2red,y4red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x2red,y4red,z4red))

       ep43=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x3red,y4red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x3red,y4red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x3red,y4red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x3red,y4red,z4red))

       ep44=((((intpointz-z2)*(intpointz-z3)*(intpointz-z4))/((z1-z2)*(z1-z3)*(z1-z4)))*masterkarray(x4red,y4red,z1red))+ & 
&        ((((intpointz-z1)*(intpointz-z3)*(intpointz-z4))/((z2-z1)*(z2-z3)*(z2-z4)))*masterkarray(x4red,y4red,z2red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z4))/((z3-z1)*(z3-z2)*(z3-z4)))*masterkarray(x4red,y4red,z3red))+ & 
&        ((((intpointz-z1)*(intpointz-z2)*(intpointz-z3))/((z4-z1)*(z4-z2)*(z4-z3)))*masterkarray(x4red,y4red,z4red))

       ep1=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*ep11)+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*ep12)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*ep13)+ & 

&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*ep14)

       ep2=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*ep21)+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*ep22)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*ep23)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*ep24)

       ep3=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*ep31)+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*ep32)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*ep33)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*ep34)

       ep4=((((intpointx-x2)*(intpointx-x3)*(intpointx-x4))/((x1-x2)*(x1-x3)*(x1-x4)))*ep41)+ & 
&        ((((intpointx-x1)*(intpointx-x3)*(intpointx-x4))/((x2-x1)*(x2-x3)*(x2-x4)))*ep42)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x4))/((x3-x1)*(x3-x2)*(x3-x4)))*ep43)+ & 
&        ((((intpointx-x1)*(intpointx-x2)*(intpointx-x3))/((x4-x1)*(x4-x2)*(x4-x3)))*ep44)

       pinterpolation=((((intpointy-y2)*(intpointy-y3)*(intpointy-y4))/((y1-y2)*(y1-y3)*(y1-y4)))*ep1)+ & 
&        ((((intpointy-y1)*(intpointy-y3)*(intpointy-y4))/((y2-y1)*(y2-y3)*(y2-y4)))*ep2)+ & 
&        ((((intpointy-y1)*(intpointy-y2)*(intpointy-y4))/((y3-y1)*(y3-y2)*(y3-y4)))*ep3)+ & 
&        ((((intpointy-y1)*(intpointy-y2)*(intpointy-y3))/((y4-y1)*(y4-y2)*(y4-y3)))*ep4)

      end if
     end if
    end if
 END FUNCTION pinterpolation

! ******************************************************************************

 SUBROUTINE sliceext(twhichslice,torbnum,sliceelectronarea,sliceholearea)
  INTEGER(KIND=int8byte), INTENT(IN) :: twhichslice,torbnum
  DOUBLE PRECISION, INTENT(OUT) :: sliceelectronarea,sliceholearea

  integer(KIND=int8byte) :: ti
  integer(KIND=int8byte) :: tj
  double precision::dintpointx
  double precision::dintpointy
  double precision::dintpointz
  double precision,allocatable,dimension(:,:)::kslice
  logical,allocatable,dimension(:,:)::unchecked

  integer(KIND=int8byte)::slicex
  integer(KIND=int8byte)::slicey
  double precision::p1
  double precision::p2
  double precision::p3
  double precision::fintpointx
  double precision::fintpointy
  double precision::fintpointz
  double precision::subtractorx
  double precision::subtractory
  double precision::subtractorz
  integer(KIND=int8byte)::nextktopeekx
  integer(KIND=int8byte)::nextktopeeky
  double precision,allocatable,dimension(:)::fsxred
  double precision,allocatable,dimension(:)::fsyred
  double precision,allocatable,dimension(:)::fsxext
  double precision,allocatable,dimension(:)::fsyext
  double precision,allocatable,dimension(:)::fsinsidexext
  double precision,allocatable,dimension(:)::fsinsideyext
  double precision,allocatable,dimension(:)::dedk
  double precision,allocatable,dimension(:)::absdedk
  double precision,allocatable,dimension(:)::einside
  integer(KIND=int8byte)::currentkx
  integer(KIND=int8byte)::currentky
  integer(KIND=int8byte)::formerkx
  integer(KIND=int8byte)::formerky
  integer(KIND=int8byte)::initialkx
  integer(KIND=int8byte)::initialky
  integer(KIND=int8byte)::currentglancekx
  integer(KIND=int8byte)::currentglanceky
  integer(KIND=int8byte)::firstfindkx
  integer(KIND=int8byte)::firstfindky
  integer(KIND=int8byte)::numglances
  logical::continueglancing
  logical::firstwarning
  logical::continuestepping
  logical::openorbit
  integer(KIND=int8byte)::nnNx
  integer(KIND=int8byte)::nnNy
  integer(KIND=int8byte)::nnEx
  integer(KIND=int8byte)::nnEy
  integer(KIND=int8byte)::nnSx
  integer(KIND=int8byte)::nnSy
  integer(KIND=int8byte)::nnWx
  integer(KIND=int8byte)::nnWy
  integer(KIND=int8byte)::numpointsonfs
  integer(KIND=int8byte)::howmanyfs
  integer(KIND=int8byte)::numopenorbs
  integer(KIND=int8byte)::numsmallorbs
  double precision::fsdAde
  double precision::fsarea
  double precision::fsinsidearea
  double precision::fsavgx
  double precision::fsavgy
  double precision::fsstdx
  double precision::fsstdy
  double precision::fsmaxx
  double precision::fsmaxy
  double precision::fsminx
  double precision::fsminy
  double precision::fsmstar
  double precision::fsdhvafrequency
  double precision::orbittype
  logical::hitsupercellborder
  logical::toosmall
  character(len=1)::slicepoint
  character(len=scarraydimension)::sliceline

 allocate(kslice(scarraydimension,scarraydimension),unchecked(scarraydimension,scarraydimension))
 allocate(fsxred(slicedimension),fsyred(slicedimension),fsxext(slicedimension))
 allocate(fsyext(slicedimension),fsinsidexext(slicedimension),fsinsideyext(slicedimension))
 allocate(dedk(slicedimension),absdedk(slicedimension),einside(slicedimension))

  do slicex=1,numx
   do slicey=1,numy

!***XXX***XXX***XXX
! Here I do the k-grid interpolation, for any direction of H-vector

! XXXXXXXXXXXXXX change the lines below to change where the supercell is placed relative to the RUC
! (but any change here needs to be mirrored in the orbit avg coord SC-to-RUC translation code near
! the end of the main code block
    p1=maxlreciplat*(((((dble(slicex)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(p*p*u + c))-((((dble(slicey)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(p*q*u))+((((dble(twhichslice)-1.0D0)/(dble(numint)-1.0D0))-1.0D0)*(q*s)))
    p2=maxlreciplat*(((((dble(slicex)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(-p*q*u))+((((dble(slicey)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(q*q*u + c))+((((dble(twhichslice)-1.0D0)/(dble(numint)-1.0D0))-1.0D0)*(p*s)))
    p3=maxlreciplat*(((((dble(slicex)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(-q*s))-((((dble(slicey)-1.0D0)/(dble(numint)-1.0D0))- &
&1.0D0)*(p*s))+((((dble(twhichslice)-1.0D0)/(dble(numint)-1.0D0))-1.0D0)*(c)))

    fintpointx=((((ai*p1)-(aii*p2)+(aiii*p3))/bigd)*(dble(nx)-1.0D0))
    fintpointy=((((bi*p1)-(bii*p2)+(biii*p3))/bigd)*(dble(ny)-1.0D0))
    fintpointz=((((ci*p1)-(cii*p2)+(ciii*p3))/bigd)*(dble(nz)-1.0D0))

    subtractorx=((floor(fintpointx/(dble(nx)-1.0D0)))*(dble(nx)-1.0D0))
    subtractory=((floor(fintpointy/(dble(ny)-1.0D0)))*(dble(ny)-1.0D0))
    subtractorz=((floor(fintpointz/(dble(nz)-1.0D0)))*(dble(nz)-1.0D0))

    dintpointx=fintpointx-subtractorx+1.0D0
    dintpointy=fintpointy-subtractory+1.0D0
    dintpointz=fintpointz-subtractorz+1.0D0

    kslice(slicex,slicey)=pinterpolation(dintpointx,dintpointy,dintpointz)
    unchecked(slicex,slicey)=.TRUE.
   end do
  end do

  if (hvd/='r') then
   do tj=1,numy
    if (tj==1) then
     sliceline='0'
    else if (tj==nint(1.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='9'
    else if (tj==nint(2.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='8'
    else if (tj==nint(3.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='7'
    else if (tj==nint(4.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='6'
    else if (tj==nint(5.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='5'
    else if (tj==nint(6.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='4'
    else if (tj==nint(7.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='3'
    else if (tj==nint(8.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='2'
    else if (tj==nint(9.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline='1'
    else if (tj==numy) then
     sliceline='0'
    else
     sliceline='*'
    end if
    do ti=1,numx
     if (kslice(ti,numy-tj+1)>fermienergy) then
      slicepoint='+'
     else if (kslice(ti,numy-tj+1)<fermienergy) then
      slicepoint='-'
     else
      slicepoint='@'
     end if
     sliceline=sliceline(1:len_trim(sliceline))//slicepoint
    end do
    if (tj==1) then
     sliceline=sliceline(1:len_trim(sliceline))//'0'
    else if (tj==nint(1.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'9'
    else if (tj==nint(2.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'8'
    else if (tj==nint(3.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'7'
    else if (tj==nint(4.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'6'
    else if (tj==nint(5.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'5'
    else if (tj==nint(6.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'4'
    else if (tj==nint(7.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'3'
    else if (tj==nint(8.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'2'
    else if (tj==nint(9.0D0*((dble(numy)+1.0D0)/10.0D0))) then
     sliceline=sliceline(1:len_trim(sliceline))//'1'
    else if (tj==numy) then
     sliceline=sliceline(1:len_trim(sliceline))//'0'
    else
     sliceline=sliceline(1:len_trim(sliceline))//'*'
    end if
    slicepic(twhichslice,tj)=sliceline
   end do
  end if

  !----------------------------
  ! Main Algorithm
  !

  nextktopeekx=1
  nextktopeeky=1
  howmanyfs=0
  numopenorbs=0
  numsmallorbs=0
  sliceelectronarea=0.0D0
  sliceholearea=0.0D0
  hitsupercellborder=.FALSE.
  openorbit=.FALSE.

  do while(nextktopeeky<=numy)

   if (unchecked(nextktopeekx,nextktopeeky).AND.kslice(nextktopeekx,nextktopeeky)<=fermienergy) then

    initialkx=nextktopeekx
    initialky=nextktopeeky
    currentkx=nextktopeekx
    currentky=nextktopeeky
    formerkx=nextktopeekx
    formerky=nextktopeeky
    numglances=0

    numpointsonfs=0
    hitsupercellborder=.FALSE.

    nnNx=currentkx
    nnNy=currentky+1
    nnEx=currentkx+1
    nnEy=currentky
    nnSx=currentkx
    nnSy=currentky-1
    nnWx=currentkx-1
    nnWy=currentky

    if (currentky==numy) then
     hitsupercellborder=.TRUE.
     nnNy=numy
    end if

    if (currentky==1) then
     hitsupercellborder=.TRUE.
     nnSy=1
    end if

    if (currentkx==numx) then
     hitsupercellborder=.TRUE.
     nnEx=numx
    end if

    if (currentkx==1) then
     hitsupercellborder=.TRUE.
     nnWx=1
    end if
   
    if ((hitsupercellborder.eqv..FALSE.).AND.((kslice(nnNx,nnNy)>fermienergy).OR. &
&(kslice(nnSx,nnSy)>fermienergy).OR.(kslice(nnEx,nnEy)>fermienergy).OR.(kslice(nnWx,nnWy)>fermienergy))) then

     howmanyfs=howmanyfs+1
     continueglancing=.TRUE.
     continuestepping=.TRUE.
     firstwarning=.FALSE.

     do ti=1,5
      if (continueglancing.eqv..TRUE.) then
       if ((ti==1).OR.(ti==5)) then
        currentglancekx=nnEx
        currentglanceky=nnEy
       end if
       if (ti==2) then
        currentglancekx=nnSx
        currentglanceky=nnSy
       end if
       if (ti==3) then
        currentglancekx=nnWx
        currentglanceky=nnWy
       end if
       if (ti==4) then
        currentglancekx=nnNx
        currentglanceky=nnNy
       end if
     
       unchecked(currentglancekx,currentglanceky)=.FALSE.
       numglances=numglances+1

       if (numpointsonfs==0) then

        if (kslice(currentglancekx,currentglanceky)>fermienergy) then
         numpointsonfs=numpointsonfs+1
         firstfindkx=currentglancekx
         firstfindky=currentglanceky
         fsxred(numpointsonfs)=dble(currentkx)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglancekx-currentkx)
         fsyred(numpointsonfs)=dble(currentky)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglanceky-currentky)

         fsxext(numpointsonfs)=(dble(currentkx-1)*xkseparation)+ &
&(dabs(kslice(currentkx,currentky)-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&          dabs(kslice(currentglancekx,currentglanceky)-fermienergy))) &
&*((dble(currentglancekx-1)*xkseparation)-(dble(currentkx-1)*xkseparation))
         fsyext(numpointsonfs)=(dble(currentky-1)*ykseparation)+(dabs(kslice(currentkx,currentky) &
&-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&          dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*((dble(currentglanceky-1) &
&*ykseparation)-(dble(currentky-1)*ykseparation))
         fsinsidexext(numpointsonfs)=dble(currentkx-1)*xkseparation
         fsinsideyext(numpointsonfs)=dble(currentky-1)*ykseparation
         dedk(numpointsonfs)=(kslice(currentglancekx,currentglanceky)-kslice(currentkx,currentky)) &
&/(((dble(currentglancekx-1)*xkseparation)- & 
&          (dble(currentkx-1)*xkseparation))+((dble(currentglanceky-1)*ykseparation)- &
&(dble(currentky-1)*ykseparation)))
         einside(numpointsonfs)=kslice(currentkx,currentky)
        end if

       else

         if (kslice(currentglancekx,currentglanceky)>fermienergy) then
          numpointsonfs=numpointsonfs+1
          fsxred(numpointsonfs)=dble(currentkx)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglancekx-currentkx)
          fsyred(numpointsonfs)=dble(currentky)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglanceky-currentky)
          fsxext(numpointsonfs)=(dble(currentkx-1)*xkseparation)+(dabs(kslice(currentkx,currentky) &
&-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&           dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))* &
&((dble(currentglancekx-1)*xkseparation)-(dble(currentkx-1)*xkseparation))
          fsyext(numpointsonfs)=(dble(currentky-1)*ykseparation)+(dabs(kslice(currentkx,currentky) &
&-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&           dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))* &
&((dble(currentglanceky-1)*ykseparation)-(dble(currentky-1)*ykseparation))
          fsinsidexext(numpointsonfs)=dble(currentkx-1)*xkseparation
          fsinsideyext(numpointsonfs)=dble(currentky-1)*ykseparation
          dedk(numpointsonfs)=(kslice(currentglancekx,currentglanceky)- &
&kslice(currentkx,currentky))/(((dble(currentglancekx-1)*xkseparation)- & 
&           (dble(currentkx-1)*xkseparation))+((dble(currentglanceky-1)*ykseparation) &
&-(dble(currentky-1)*ykseparation)))
          einside(numpointsonfs)=kslice(currentkx,currentky)
         else
          continueglancing=.FALSE.
          formerkx=currentkx
          formerky=currentky
          currentkx=currentglancekx
          currentky=currentglanceky
          continuestepping=.TRUE.
         end if
      
       end if
      end if
     end do

     if (numpointsonfs==5) then
      continuestepping=.FALSE.
     end if

     do while(continuestepping.eqv..TRUE.)

      continueglancing=.TRUE.
      numglances=0

      nnNx=currentkx
      nnNy=currentky+1
      nnEx=currentkx+1
      nnEy=currentky
      nnSx=currentkx
      nnSy=currentky-1
      nnWx=currentkx-1
      nnWy=currentky
      if (currentky==numy) then
       hitsupercellborder=.TRUE.
      end if
      if (currentky==1) then
       hitsupercellborder=.TRUE.
      end if
      if (currentkx==numx) then
       hitsupercellborder=.TRUE.
      end if
      if (currentkx==1) then
       hitsupercellborder=.TRUE.
      end if

      if (hitsupercellborder.eqv..FALSE.) then
       do ti=1,4
        if (continueglancing.eqv..TRUE.) then

         if (((currentkx>formerkx).AND.(.NOT.((formerkx==1).AND. &
&(currentkx==numx)))).OR.((formerkx==numx).AND.(currentkx==1))) then
          !coming from the left
          if ((ti==1)) then
           currentglancekx=nnNx
           currentglanceky=nnNy
          end if
          if (ti==2) then
           currentglancekx=nnEx
           currentglanceky=nnEy
          end if
          if (ti==3) then
           currentglancekx=nnSx
           currentglanceky=nnSy
          end if
          if (ti==4) then
           currentglancekx=nnWx
           currentglanceky=nnWy
          end if
         end if

         if (((currentkx<formerkx).AND.(.NOT.((formerkx==numx).AND. &
&(currentkx==1)))).OR.((formerkx==1).AND.(currentkx==numx))) then
          !coming from the right
          if ((ti==1)) then
           currentglancekx=nnSx
           currentglanceky=nnSy
          end if
          if (ti==2) then
           currentglancekx=nnWx
           currentglanceky=nnWy
          end if
          if (ti==3) then
           currentglancekx=nnNx
           currentglanceky=nnNy
          end if
          if (ti==4) then
           currentglancekx=nnEx
           currentglanceky=nnEy
          end if
         end if

         if (((currentky>formerky).AND.(.NOT.((formerky==1).AND. &
&(currentky==numy)))).OR.((formerky==numy).AND.(currentky==1))) then
          !coming from below
          if ((ti==1)) then
           currentglancekx=nnWx
           currentglanceky=nnWy
          end if
          if (ti==2) then
           currentglancekx=nnNx
           currentglanceky=nnNy
          end if
          if (ti==3) then
           currentglancekx=nnEx
           currentglanceky=nnEy
          end if
          if (ti==4) then
           currentglancekx=nnSx
           currentglanceky=nnSy
          end if
         end if

         if (((currentky<formerky).AND.(.NOT.((formerky==numy).AND. &
&(currentky==1)))).OR.((formerky==1).AND.(currentky==numy))) then
          !coming from above
          if ((ti==1)) then
           currentglancekx=nnEx
           currentglanceky=nnEy
          end if
          if (ti==2) then
           currentglancekx=nnSx
           currentglanceky=nnSy
          end if
          if (ti==3) then
           currentglancekx=nnWx
           currentglanceky=nnWy
          end if
          if (ti==4) then
           currentglancekx=nnNx
           currentglanceky=nnNy
          end if
         end if

          unchecked(currentglancekx,currentglanceky)=.FALSE.
          numglances=numglances+1

          if (kslice(currentglancekx,currentglanceky)>fermienergy) then
           numpointsonfs=numpointsonfs+1
           fsxred(numpointsonfs)=dble(currentkx)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglancekx-currentkx)
           fsyred(numpointsonfs)=dble(currentky)+(dabs(kslice(currentkx,currentky)- &
&fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ &
&dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*dble(currentglanceky-currentky)
           fsxext(numpointsonfs)=(dble(currentkx-1)*xkseparation)+ &
&(dabs(kslice(currentkx,currentky)-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&            dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))* &
&((dble(currentglancekx-1)*xkseparation)-(dble(currentkx-1)*xkseparation))
           fsyext(numpointsonfs)=(dble(currentky-1)*ykseparation)+ &
&(dabs(kslice(currentkx,currentky)-fermienergy)/(dabs(kslice(currentkx,currentky)-fermienergy)+ & 
&            dabs(kslice(currentglancekx,currentglanceky)-fermienergy)))*((dble(currentglanceky-1) &
&*ykseparation)-(dble(currentky-1)*ykseparation))
           fsinsidexext(numpointsonfs)=dble(currentkx-1)*xkseparation
           fsinsideyext(numpointsonfs)=dble(currentky-1)*ykseparation
           dedk(numpointsonfs)=(kslice(currentglancekx,currentglanceky)- &
&kslice(currentkx,currentky))/(((dble(currentglancekx-1)*xkseparation)- & 
&            (dble(currentkx-1)*xkseparation))+((dble(currentglanceky-1)* &
&ykseparation)-(dble(currentky-1)*ykseparation)))
           einside(numpointsonfs)=kslice(currentkx,currentky)

           if ((firstwarning.eqv..TRUE.).AND.(currentglancekx==firstfindkx) &
&.AND.(currentglanceky==firstfindky)) then
            continueglancing=.FALSE.
            continuestepping=.FALSE.
           end if

          else
           continueglancing=.FALSE.
           formerkx=currentkx
           formerky=currentky
           currentkx=currentglancekx
           currentky=currentglanceky
           continuestepping=.TRUE.
           firstwarning=.FALSE.
          end if       

        end if
       end do
      end if

      if ((currentkx==initialkx).AND.(currentky==initialky).AND. &
&(firstwarning.eqv..FALSE.)) then
       firstwarning=.TRUE.
      end if

      if (hitsupercellborder.eqv..TRUE.) then
       openorbit=.TRUE.
       continuestepping=.FALSE.
       continueglancing=.FALSE.
      else
       openorbit=.FALSE.
      end if

     end do

     if (openorbit.eqv..TRUE.) then
      howmanyfs=howmanyfs-1
      numopenorbs=numopenorbs+1
     else

      fsarea=0.0D0
      fsinsidearea=0.0D0
      do ti=1,(numpointsonfs-1)
       fsarea=fsarea+(((fsxext(ti))*(fsyext(ti+1)))-((fsxext(ti+1))*(fsyext(ti))))
       fsinsidearea=fsinsidearea+(((fsinsidexext(ti))* &
&(fsinsideyext(ti+1)))-((fsinsidexext(ti+1))*(fsinsideyext(ti))))
      end do
      fsarea=fsarea/2.0D0
      fsinsidearea=fsinsidearea/2.0D0
      fsarea=dabs(fsarea)
      fsinsidearea=dabs(fsinsidearea)
      fsdhvafrequency=fsarea*convfsarea2kt
      if (fsarea<minarea) then
       toosmall=.TRUE.
       howmanyfs=howmanyfs-1
       numsmallorbs=numsmallorbs+1
      else
       toosmall=.FALSE.
      end if

!-- start block of code
      do ti=1,(numpointsonfs-1)

       if (((fsxred(ti)==dnint(fsxred(ti))).AND.(fsxred(ti+1)==dnint(fsxred(ti+1)))) &
&.OR.((fsyred(ti)==dnint(fsyred(ti))).AND.(fsyred(ti+1)==dnint(fsyred(ti+1))))) then
        if ((fsxred(ti)==dnint(fsxred(ti))).AND.(fsxred(ti+1)==dnint(fsxred(ti+1)))) then
!----------horizontal line segment case
         absdedk(ti)=dsqrt(((dedk(ti))*(dedk(ti)))+(((einside(ti+1)-einside(ti))/xkseparation) &
&*((einside(ti+1)-einside(ti))/xkseparation)))
        else
!----------vertical line segment case
         absdedk(ti)=dsqrt(((dedk(ti))*(dedk(ti)))+(((einside(ti+1)-einside(ti))/ykseparation) &
&*((einside(ti+1)-einside(ti))/ykseparation)))
        end if
       else
        absdedk(ti)=dsqrt(((dedk(ti))*(dedk(ti)))+((dedk(ti+1))*(dedk(ti+1))))
       end if
      end do
      fsdAde=0.0D0
      do ti=1,(numpointsonfs-1)
       fsdAde=fsdAde+(1.0D0/absdedk(ti))*(dsqrt(((fsxext(ti+1)-fsxext(ti))*(fsxext(ti+1)- &
&fsxext(ti)))+((fsyext(ti+1)-fsyext(ti))*(fsyext(ti+1)-fsyext(ti)))))
      end do
!-- end block of code

      fsmstar=fsdAde*convfsdade2mstar

      if (fsinsidearea==fsarea) then
       orbittype=0.0D0
      else
       if (fsinsidearea>fsarea) then
        orbittype=-1.0D0
        sliceholearea=sliceholearea+fsarea
       end if
       if (fsinsidearea<fsarea) then
        orbittype=1.0D0
        sliceelectronarea=sliceelectronarea+fsarea
       end if
      end if

      fsavgx=0.0D0
      fsavgy=0.0D0
      fsstdx=0.0D0
      fsstdy=0.0D0
      fsmaxx=fsxext(1)
      fsmaxy=fsyext(1)
      fsminx=fsxext(1)
      fsminy=fsyext(1)
      do ti=1,(numpointsonfs-1)
       fsavgx=fsavgx+fsxext(ti)
       fsavgy=fsavgy+fsyext(ti)
      end do
      do ti=2,(numpointsonfs-1)
       if (fsxext(ti)>fsmaxx) then
        fsmaxx=fsxext(ti)
       end if
       if (fsyext(ti)>fsmaxy) then
        fsmaxy=fsyext(ti)
       end if
       if (fsxext(ti)<fsminx) then
        fsminx=fsxext(ti)
       end if
       if (fsyext(ti)<fsminy) then
        fsminy=fsyext(ti)
       end if
      end do
      fsavgx=fsavgx/dble(numpointsonfs-1)
      fsavgy=fsavgy/dble(numpointsonfs-1)
      do ti=1,(numpointsonfs-1)
       fsstdx=fsstdx+((fsxext(ti)-fsavgx)*(fsxext(ti)-fsavgx))
       fsstdy=fsstdy+((fsyext(ti)-fsavgy)*(fsyext(ti)-fsavgy))
      end do
      fsstdx=dsqrt(fsstdx/(numpointsonfs-1)) ! This is a "population" standard deviation
      fsstdy=dsqrt(fsstdy/(numpointsonfs-1)) ! This is a "population" standard deviation

      fsavgx=fsavgx/xlength
      fsavgy=fsavgy/ylength
      fsstdx=fsstdx/xlength
      fsstdy=fsstdy/ylength
      fsmaxx=fsmaxx/xlength
      fsmaxy=fsmaxy/ylength
      fsminx=fsminx/xlength
      fsminy=fsminy/ylength

      if (toosmall.eqv..FALSE.) then
       numslfs(twhichslice)=howmanyfs
       numsltoosmall(twhichslice)=numsmallorbs
       numslopen(twhichslice)=numopenorbs
       slfsarea(twhichslice,howmanyfs)=fsarea
       slfsmstar(twhichslice,howmanyfs)=fsmstar
       slfsfreq(twhichslice,howmanyfs)=fsdhvafrequency
       slfsorbtype(twhichslice,howmanyfs)=orbittype
       slfsorbnum(twhichslice,howmanyfs)=howmanyfs
       slfsavgx(twhichslice,howmanyfs)=fsavgx
       slfsavgy(twhichslice,howmanyfs)=fsavgy
       slfsstdx(twhichslice,howmanyfs)=fsstdx
       slfsstdy(twhichslice,howmanyfs)=fsstdy
       slfsmaxx(twhichslice,howmanyfs)=fsmaxx
       slfsmaxy(twhichslice,howmanyfs)=fsmaxy
       slfsminx(twhichslice,howmanyfs)=fsminx
       slfsminy(twhichslice,howmanyfs)=fsminy
       if (howmanyfs==torbnum) then
        noutlinepts=numpointsonfs-1
        outlineavgxext=fsavgx
        outlineavgyext=fsavgy
        outlinefreq=fsdhvafrequency
        do ti=1,(numpointsonfs-1)
         fsoutlinexext(ti)=fsxext(ti)
         fsoutlineyext(ti)=fsyext(ti)
        end do
       end if
      end if

     end if

    end if

   end if

   unchecked(nextktopeekx,nextktopeeky)=.FALSE.
   if (nextktopeekx==numx) then
    nextktopeekx=1
    nextktopeeky=nextktopeeky+1
   else
    nextktopeekx=nextktopeekx+1
   end if

  end do

 END SUBROUTINE sliceext

! ******************************************************************************

 SUBROUTINE pgettime(tmphr,tmpmin,tmpsec,tmpms)
  INTEGER(KIND=int8byte), INTENT(INOUT) :: tmphr,tmpmin,tmpsec,tmpms
  CALL DATE_AND_TIME(date=pdate,time=ptime,zone=ptimezone,values=ptimedatevalues)
  tmphr=ptimedatevalues(5)
  tmpmin=ptimedatevalues(6)
  tmpsec=ptimedatevalues(7)
  tmpms=ptimedatevalues(8)
 END SUBROUTINE pgettime

! ******************************************************************************

 SUBROUTINE ptimediff(tmphr,tmpmin,tmpsec,tmpms)
  INTEGER(KIND=int8byte), INTENT(INOUT) :: tmphr,tmpmin,tmpsec,tmpms
  CALL DATE_AND_TIME(date=pdate,time=ptime,zone=ptimezone,values=ptimedatevalues)
  tmphr=ptimedatevalues(5)-tmphr
  tmpmin=ptimedatevalues(6)-tmpmin
  tmpsec=ptimedatevalues(7)-tmpsec
  tmpms=ptimedatevalues(8)-tmpms
  if (tmpms<0) then
   tmpms=tmpms+1000
   tmpsec=tmpsec-1
  end if
  if (tmpsec<0) then
   tmpsec=tmpsec+60
   tmpmin=tmpmin-1
  end if
  if(tmpmin<0) then
   tmpmin=tmpmin+60
   tmphr=tmphr-1
  end if
 END SUBROUTINE ptimediff

! ******************************************************************************

END PROGRAM skeaf
