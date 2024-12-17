***********************************************************************************************************
*                                                                                                         *
* README file for Supercell K-space Extremal Area Finder (SKEAF) version 1.3.0 release build 149          *
*                                                                                                         *
* Copyright (C) 2012 Patrick Rourke (rourke@gmail.com)                                                    *
*                                                                                                         *
* Algorithm publication -- P.M.C. Rourke and S.R. Julian, Computer Physics Communications 183, 324 (2012) *
* Algorithm build date -- 29 July, 2012                                                                   *
* Readme last revision date -- 29 July, 2012                                                              *
*                                                                                                         *
*---------------------------------------------------------------------------------------------------------*
*                                                                                                         *
* This program is free software: you can redistribute it and/or modify it under the terms of the GNU      *
* General Public License as published by the Free Software Foundation, either version 3 of the License,   *
* or (at your option) any later version.                                                                  *
*                                                                                                         *
* This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even  *
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General       *
* Public License for more details.                                                                        *
*                                                                                                         *
* You should have received a copy of the GNU General Public License along with this program               *
* (default filename is COPYING.txt).  If not, see <http://www.gnu.org/licenses/>.                         *
*                                                                                                         *
**********************************************************************************************************


1. CHANGELOG
2. INTRODUCTION
3. COMPILATION
4. PROGRAM USAGE
5. CHECKING OUTPUT
6. MISCELLANEOUS NOTES
7. CONTACT


1. CHANGELOG

SKEAF v1.3.0 r149 (Sunday, 29 July, 2012)
(Thanks to Pascal Reiss, Sven Friedemann and Gianni Profeta for useful discussions and feedback.)
-Further improved the orbit matching routine to reduce spurious frequencies coming from mismatches across Fermi surface bifurcations
--> Orbits on adjacent slices must satisfy the criteria for matching (section 2.5 of Rourke & Julian CPC 2012 article) in both forward (bottom to top of super cell) and reverse (top to bottom of super cell) directions in order to be matched to the same Fermi surface chunk
--> In order for orbits on adjacent slices to be matched to the same Fermi surface chunk both orbits' average coordinates must be within the other orbit's bounding box (defined by max & min coordinates)
--> Attempt to detect Fermi surface bifurcations (and eliminate matching across them) by comparing multiple orbit bounding box overlaps
--> Extremal orbits whose maximum or minimum coordinates are within two standard deviations of the super-cell walls can be excluded from the output (controlled by the user); this eliminates false matches across bifurcations when one arm of the bifurcation hits the super-cell wall (see Section 4 below)
-Coordinates of extremal orbit outlines are now available in the results_orbitoutlines files (see Section 4 below; based on a contribution by Pascal Reiss at the University of Cambridge)
-Added a section ("5. CHECKING OUTPUT") to this README file describing how to check the validity of SKEAF output; the sections formerly known as "5." and "6." are now "6." and "7.", respectively
-Clarified in the SKEAF output that the unit cell volume calculated by SKEAF is that of the Reciprocal Unit Cell (RUC), not the real-space crystal unit cell
-Clarified in the SKEAF output that the DOS(E_F) calculated by SKEAF is per spin direction (see notes in Section 6 below about how to compare SKEAF's DOS(E_F) to other measures of the density of states)
-Updated the spheroid simbxsf program to smooth out the energy discontinuity across RUC edges when the surface is shifted away from the RUC centre

SKEAF v1.2.0 r124 (Wednesday, 16 May, 2012)
(Thanks to Pascal Reiss, David Billington and Jian-Xin Zhu for useful discussions and feedback.)
-Added a note to this README file about problems with using ELK/exciting to generate BXSF files (see Section 4 below)
-Included the Fortran 90 program "ELK_exciting_BXSFconverter" to help correct problems with ELK/exciting BXSF files
-Improved the orbit matching routine so that it no longer makes the mismatches that were leading to spurious scattered frequencies in previous versions of SKEAF
--> Due to the efficacy of this improvement, the ability to automatically exclude orbits with few copies found in the super cell (added in SKEAF v1.1.0 r104) is not longer required and has been removed both from the code and from Section 4 of this README file; the number of copies found for each orbit is still listed in the output, so users may still remove rare orbits manually if required
-Input BXSF filename can now optionally be specified via the command line option -fbxsf (see Section 4 below)
-BXSF file read subroutine is now less strict about number of energies per line in the input file, and better at communicating the cause when certain BXSF read operations fail
-Standard deviations listed as estimates of the error bounds in the final averaged quantities (Freq., m*, etc.) have been changed from population standard deviations to sample standard deviations; standard deviations of the points on the orbit outlines remain population standard deviations (see note in Section 5 below)
-Simplified the interpolation parameter input procedure
-Simplified some of the output displayed on the screen and in results_short.out (full information can still be found in results_long.out)
-Clarified the SKEAF parameter on-screen prompts to indicate that more interpolation gives more accurate results
-Changed the SKEAF parameter on-screen prompts to suggest a default "Maximum distance (fraction of RUC side length) between average coordinates for orbit averaging? [0-1]" value of 0.05 (rather than the previous 0.1), for consistency with the Rourke & Julian CPC 2012 article
-Clarified the SKEAF parameter on-screen prompts to indicate that "Maximum distance (fraction of RUC side length) between average coordinates for orbit averaging? [0-1]" is usually chosen to be 1 when running in auto-rot. mode
-Corrected band electron-like/hole-like estimation in results_long.out for each angle when running in auto-rot. mode
-Freq. and m* are now expressed in exponential notation in the results_freqvsangle.out file
-Added an explicit initialization of the 'openorbit' variable, in order to avoid occasional crashes if SKEAF is compiled with the ifort compiler option -check (though using the -check compiler option is generally not recommended)
-Clarified the difference between maxnuminput and maxnumint in the "memoryestimateskeaf" program and this README file (see Section 3 below)
-Simplified the code in the simbxsf programs to provide a clearer example of how to make BXSF files; these programs now ask the user for the real space simple cubic lattice constant, and use that to build the RUC
-Added a note to this README file about 64-bit ifort compilation on Mac OSX systems (see Section 3 below)

SKEAF v1.1.0 r104 (Tuesday, 03 April, 2012)
-Memory usage dramatically reduced
-Added OpenMP directives and re-worked code for parallel code execution
-Fermi surface curvature near extremal orbits now reported (see note in Section 5 below)
-Orbits with few copies found in the super cell can now be excluded from output (see note #4 in Section 4 below)

SKEAF v1.0.0 r091 (Thursday, 22 March, 2012)
-First official public release of the SKEAF algorithm!
-GNU GPL information added


2. INTRODUCTION

The Supercell K-space Extremal Area Finder (SKEAF) program allows extremal electron/hole Fermi surface orbits to be extracted from a set of band energies, usually generated by a Density Functional Theory (DFT) package.

Starting from the set of calculated band energies, SKEAF constructs a large k-space "super cell," and uses this to determine quantum oscillation (e.g. dHvA) frequencies, band masses, orbit types (electron or hole), local Fermi surface curvatures, and band density of states contributions. Refer to the article P.M.C. Rourke and S.R. Julian, Computer Physics Communications 183, 324 (2012) (also available at http://arxiv.org/abs/0803.1895) for a detailed introduction to and explanation of the algorithm underlying this program.

Please cite the Rourke & Julian, Comput. Phys. Commun. 183, 324 (2012) paper in any publications making use of SKEAF.


3. COMPILATION

The SKEAF source code is written in the Fortran 90 language, and thus requires a Fortran 90 compiler to produce a machine-executable compiled program version. Development and testing of the program has proceeded primarily under Ubuntu Linux using the Intel Fortran Compiler "ifort" -- thus the ifort compiler is recommended for optimal user experience. This compiler is available to download directly from Intel at no cost for non-commercial use. Some success has also been reported with other compilers such as gfortran under Ubuntu Linux and g95 under Microsoft Windows 7 and Microsoft Windows XP, though they have not been extensively tested with the SKEAF codebase and tend to produce considerably slower executables than ifort.

If the program compiles properly, but fails to run (often with a message of "Killed" in the Linux shell), the declared value of the "maxnumint" parameter in the source code must be decreased from its default of 150. Maxnumint sets the maximum number of points that can be interpolated per single cell side -- the full super cell will have sides that are 4x longer than the "single cell side", and therefore will have 4x as many points per side. Since array sizes, and therefore total memory usage, scale roughly like the square of 4*maxnumint, large values of maxnumint may cause SKEAF to demand more memory than is physically present on the machine, causing the error. Since large super-cell interpolation values give more accurate results, it is important that maxnumint be as large as reasonably possible. Thus, gradually decrease the declared value of maxnumint in the source code, re-compiling after each change, until the program runs without being immediately killed. It may also help to have the Linux stack size set to "unlimited" (ulimit -s unlimited). Conversely, many systems can handle maxnumint values larger than 150.

The Fortran 90 program "memoryestimateskeaf" has been included to provide a guide for choosing an appropriate value of maxnumint. Note that "memory" means AVAILABLE memory at SKEAF runtime, not total memory. Furthermore, although SKEAF can use all of the available RAM + swap space as its working memory, dipping into hard disk drive-based swap space will result in the program running very slowly. Also, keep in mind when working with the "memoryestimateskeaf" program that maxnuminput is not the same as maxnumint -- just leave maxnuminput at 100 unless you will be reading BXSF files with more than 100 points per side.

Memory addressing limitations on 32-bit systems mean that such systems can use no more than 2 GB of available memory. For 32-bit Linux systems with recent versions of the Intel Fortran Compiler, the following compiler options are recommended for optimal speed without sacrificing numerical accuracy:
-xHost -O3 -ipo

On systems with a 64-bit processor + 64-bit operating system + 64-bit Intel Fortran Compiler, maxnumint may be increased to values requiring more than 2 GB of available memory. For such 64-bit systems, the following compiler options should be used:
-mcmodel=medium -shared-intel -xHost -O3 -ipo

On 64-bit Mac OSX systems running the Intel Fortran Compiler, the -m64 compiler option should be used instead of -mcmodel=medium -shared-intel.

Parallel execution of SKEAF on shared-memory parallel architecture (e.g. multi-core) systems is supported via OpenMP (http://openmp.org/). To enable this, make sure that OpenMP is set up on your system (i.e. OMP_NUM_THREADS environment variable is set), then include the option -openmp when compiling SKEAF with ifort (-fopenmp in gfortran). For 32-bit multi-core ifort systems the following compiler options are recommended:
-xHost -O3 -ipo -openmp

For 64-bit multi-core ifort systems the following compiler options are recommended:
-mcmodel=medium -shared-intel -xHost -O3 -ipo -openmp

The author typically uses a 64-bit dual-core system with maxnumint = 200 and 2 OpenMP threads.


4. PROGRAM USAGE

Once SKEAF has been compiled, the compiled executable file should be run, and the on-screen prompts followed.

Additionally, several optional command line arguments may be included at runtime: -rdcfg tells SKEAF to automatically read input parameters from the config.in file (this file is generated when one enters parameters manually), -nodos tells SKEAF to skip the density of states calculation, -noext tells SKEAF to skip the extremal area finding portion of the algorithm, and -fbxsf filename.bxsf (where "filename.bxsf" would be replaced with the actual name of the file you want to read) tells SKEAF to use the listed BXSF filename. In combination with Linux scripting, these command line flags are useful for instructing the program to automatically run through several bands, for example.

As described in the article Rourke & Julian, Comput. Phys. Commun. 183, 324 (2012), SKEAF reads band energies stored in files formatted according to the XCrysDen BXSF specification (see http://www.xcrysden.org/doc/XSF.html#__toc__14 for details). These need to be calculated using an external density functional theory code package, and each BXSF file must contain only one band. Reciprocal lattice vectors must be in units of 1/a.u. and must not contain the factor of 2*Pi (e.g. reciprocal lattice vectors should be of the form 1/lattconst, not 2*Pi/lattconst); and the Fermi and band energies must be in units of Rydbergs. For testing purposes, the included Fortran 90 programs "simbxsf_sphere", "simbxsf_spheroid", "simbxsf_cylinder", and "simbxsf_barrel" allow the user to generate idealized Fermi surface BXSF files without running any density functional calculations; these programs can also serve as guides for how to translate your own band energies into the BXSF format.

BXSF files generated using the Wien2k+XCrysDen combination (http://www.wien2k.at/ & http://www.xcrysden.org/) generally work in SKEAF without any further changes. BXSF files generated with Density Functional Theory packages other than Wien2k might not conform to the specifications outlined above, and therefore might lead to incorrect results when analyzed by SKEAF. SKEAF has no way of knowing where the BXSF file came from and, e.g., what units are being used, so please be aware of the nature of your calculations.

As of the time of this writing, BXSF files generated using the ELK code (version 1.4.22; http://elk.sourceforge.net/) present 3 problems for SKEAF:
-the energies are specified on an incorrect Periodic Grid, rather than a General Grid
-reciprocal lattice vectors include the factor of 2*Pi
-energies are in units of Hartrees, rather than Rydbergs
Thus, for ELK users who would like to use SKEAF, the following steps should be followed:
#1. Generate a BXSF file in ELK using task 102.
#2. Open the generated BXSF file in XCrysDen, and save each band as a separate BXSF file (ELK tends to include multiple bands in the generated BXSF file, even when only one band crosses the Fermi energy).
#3. Use the included Fortran 90 program "ELK_exciting_BXSFconverter" on each band's BXSF file in order to correct the three problems listed above. The resulting BXSF files are then ready to be analyzed by SKEAF. Note that you should be sure of what problems are present and need to be corrected -- for example, if a future version of ELK eliminates some of the above listed problems, trying to correct them when they are not present will introduce new problems into the BXSF file.

As of the time of this writing, BXSF files generated using the exciting code (helium-3 isotope release; http://exciting-code.org/) present numerous problems for SKEAF:
-the energies are specified on an incorrect Periodic Grid, rather than a General Grid
-reciprocal lattice vectors include the factor of 2*Pi
-energies are in units of Hartrees, rather than Rydbergs
-band energies have been multiplied by a factor 10^(-minexp), where minexp changes from calculation to calculation
-the sign (+ve/-ve relative to the 0 of the Fermi energy) of the band energies has been switched
-another mysterious problem with the band energies that causes incorrect DOS and effective mass values to be obtained, but does not affect the obtained dHvA frequencies (please contact the author at rourke@gmail.com if you have insight into the nature of this problem!)
Thus, for exciting users who would like to use SKEAF, the following steps should be followed:
#1. Generate a BXSF file using exciting and the xmlfermis2bxsf.xsl template.
#2. If there are multiple bands in the generated BXSF file, open it in XCrysDen, and save each band as a separate BXSF file.
#3. Use the included Fortran 90 program "ELK_exciting_BXSFconverter" on each band's BXSF file in order to correct the problems listed above. The resulting BXSF files are then *somewhat* ready to be analyzed by SKEAF. Note that unless you know minexp for your specific calculation, the guess made by "ELK_exciting_BXSFconverter" may be wrong.
***IMPORTANT: EVEN AFTER CORRECTING THE MAIN PROBLEMS LISTED ABOVE, THE "MYSTERIOUS" FINAL PROBLEM IS STILL PRESENT, SO EXCITING DENSITY OF STATES CONTRIBUTIONS AND EFFECTIVE MASSES SHOULD NOT BE TRUSTED!***

During a SKEAF run, after the super-cell interpolation and magnetic field direction have been selected, the user must set four more parameters that control how the algorithm handles the extremal orbits that it detects:

#1. "Minimum allowed extremal Fermi surface dHvA frequency (in kT)?"
--> Extremal orbits with frequency smaller than this number are ignored by SKEAF. Useful for cutting out small orbits coming from one or two isolated k-points in the input file; orbits this small sometimes arise from Fermi surface corrugation effects and are usually not accurately extracted anyway. An input value of 0 ensures that no frequencies are ignored.

#2. "Maximum fractional difference between frequencies for orbit averaging? [0-1]"
--> A number between 0 and 1 that dictates how close orbit frequencies need to be to one another for SKEAF to treat them as multiple copies of the same orbit and therefore average them together. The default 0.01 means that frequencies within 1% of one another are candidates for averaging. 0 disables orbit averaging, usually resulting in lots of similar orbits with slightly different frequencies being found in different parts of the super cell. And 1 allows all frequencies to be averaged together (usually a bad idea, since it will muddle together lots of very different orbits). Note that BOTH orbit frequencies (#2.) AND orbit average coordinates (#3.) need to be within the user-defined bounds for the orbits to be averaged together.

#3. "Maximum distance (fraction of RUC side length) between average coordinates for orbit averaging? [0-1]"
--> A number between 0 and 1 that dictates how close orbit average coordinates need to be to one another for SKEAF to treat them as multiple copies of the same orbit and therefore average them together. Average coordinates of an orbit roughly represent the location of the "centre" of that orbit (but generalized to handle cases where orbits may not be centrosymmetric, etc.), listed as fractions of the original input file reciprocal lattice vectors. The default 0.05 means that orbits with average coordinates differing by no more than 5% of the lengths of the reciprocal lattice vectors are candidates for averaging. 0 disables orbit averaging, usually resulting in lots of similar orbits with slightly different frequencies being found in different parts of the super cell. And 1 allows orbits from any location to be averaged together -- this setting is typically used during auto-rotation runs to prevent many duplicated frequencies from appearing at each field angle, but means that the final listed coordinates for each orbit will be meaningless and should be ignored. Note that BOTH orbit frequencies (#2.) AND orbit average coordinates (#3.) need to be within the user-defined bounds for the orbits to be averaged together.

#4. "Allow extremal orbits located near super-cell walls? [y or n]"
--> The default n means that extremal orbits whose maximum or minimum coordinates are within two standard deviations of a slice edge (i.e. a super-cell boundary) will not be reported in the output. This is a conservative measure that eliminates false frequencies arising from mismatches across Fermi surface bifurcations in which one of the bifurcation "arms" hits the edge of the super cell. Most legitimate extremal orbits will have other copies that are found away from super-cell walls. However, some legitimate large orbits that cross RUC / Brillouin zone boundaries (such as in the cylinder and barrel test surfaces at high angles shown in the Rourke & Julian CPC 2012 article) may be eliminated by this setting. Choosing y will keep all extremal orbits, including those near super-cell walls, but in this case one must be extra careful to check the validity of the SKEAF results (see Section 5 below).

Output files are results_short.out, results_long.out (which gives more detailed information during a single-angle calculation), results_freqvsangle.out, results_orbitoutlines_invAng.out and results_orbitoutlines_invau.out. The results_orbitoutlines_invAng.out and results_orbitoutlines_invau.out files give the coordinates of the largest orbit in each averaged set in units of Angstroms^-1 and a.u.^-1 respectively, translated as close to the RUC origin (Brillouin zone centre) as possible. As of the time of this writing, XCrysDen cannot natively read and plot the results_orbitoutlines files, but Pascal Reiss, currently at the University of Cambridge (pafar2@cam.ac.uk), has developed a modification to the XCrysDen code that allows orbit outlines to be plotted. It also appears that XCrysDen displays Fermi surfaces in units of Angstroms^-1 when first building them directly from a Wien2k calculation, but in units of a.u.^-1 when reading from a BXSF file.

These output files, along with the config.in file, are overwritten upon each new program run -- if given files need to be saved, they should be copied elsewhere / renamed prior to re-running the SKEAF executable.


5. CHECKING OUTPUT

It is important to make sure that the results obtained from the SKEAF output make sense in the context of the given material's Fermi surface. Do not blindly trust them -- make sure you understand what orbits have been found!

One way to check the SKEAF results is to run SKEAF at a single angle of interest and then look in the results_long.out file:
#1. Go to the end of the file to see what extremal orbits are reported.
#2. Pick one occurrence of each extremal orbit and, using the listed slice number and orbit number, locate it earlier in the file amid the output for its specific slice.
#3. Using the listed "Average (x,y) in SC fractional coordinates" as a guide, find the orbit in the ASCII picture of the slice. In this picture, a k-point with E_k > E_F is denoted by a "+", a k-point with E_k < E_F is denoted by a "-" and a k-point with E_k = E_F is denoted by a "@". The numbers along the edges of the picture give the decimal position of the k-points -- e.g. the number 7 along the edge of the picture signifies an SC fractional coordinate of 0.7.
#4. What does the orbit look like? Does it make sense?
#5. Look at the region near the same "Average (x,y) in SC fractional coordinates" on the ASCII pictures of the adjacent slices. Compare the shapes and areas of the orbits there to those of the extremal orbit. Is the extremal orbit a true extremum?

Furthermore, if you are able to plot the extremal orbit outlines (from the results_orbitoutlines files) on the Fermi surface itself, a visual inspection can help reveal if the orbits SKEAF has found are sensible.

Since the program finds all orbits that are locally extremal (even minimally so), it is sensitive to corrugation, etc. in the calculated Fermi surfaces. Fermi surface corrugation can arise from the discrete nature of the input k-mesh, so comparing the results from the same surface calculated on different k-meshes (and using different levels of interpolation) can be instructive.

Finally, a good way to familiarize yourself with SKEAF's parameters and "sanity check" that everything is working properly is to calculate and analyze the Fermi surface of a simple, well-understood metal like face-centred-cubic (FCC) copper.


6. MISCELLANEOUS NOTES

--For most accurate results, the program should be run at the highest possible interpolation (150 points per single cell side, or more if computational set-up allows). It is also important to start with a BXSF file containing band energies calculated on a fine k-mesh, on the order of 20 000 or more k-points in the reciprocal unit cell -- denser starting k-meshes generally give better results.

--The SKEAF "a," "b," and "c" directions are defined to lie along the reciprocal lattice vectors listed first, second, and third, respectively, in the input BXSF file. Since these may or may not correspond to the desired k-space symmetry directions for the magnetic field orientation, it is often advisable to define the field direction as non-principal ("n"), using phi and theta as described in the algorithm article Rourke & Julian, Comput. Phys. Commun. 183, 324 (2012).

--SKEAF is a fairly processor-intensive algorithm and a calculation at one field-angle, at high levels of interpolation, can take several minutes on a standard desktop computer; thus, auto-rotations through many angles can take on the order of hours.

--To compare the SKEAF DOS(E_F) to the value read off a Wien2k Total DOS plot at E=E_F: multiply the SKEAF DOS(E_F) value by 2 (to include both spin up and spin down; alternately, the DOS can be calculated separately for each spin direction), divide it by the Reciprocal Unit Cell volume, then repeat this and add up the results from all bands crossing the Fermi energy, converting from Rydbergs^-1 to eV^-1 as necessary.

--To compare the SKEAF DOS(E_F) to the Sommerfeld electronic specific heat coefficient gamma: multiply the SKEAF DOS(E_F) value by ((k_B)^2)/(24*pi), then multiply by 2 (to include both spin up and spin down; alternately, the DOS can be calculated separately for each spin direction), then multiply by Avogadro's number (6.02 x 10^23 atoms/mol), then multiply by the real-space unit cell volume (in Angstroms^3), then divide by the number of atoms of interest in the unit cell (e.g. 4 Cu's for FCC copper, 1 Ce for CeCoIn5, etc.), then repeat this and add up the results from all bands crossing the Fermi energy, converting from Rydbergs to Joules, etc. as necessary.

--An estimate of the local Fermi surface curvature around each extremal orbit is reported in the output files as "Curv." This fairly crude estimate is calculated from frequencies (F) as Curv. = F'' = (Fprevslice + Fnextslice - 2*Fextremalslice)/(sliceseparation^2), and therefore depends on the level of super-cell interpolation (higher interpolation will probe the curvature closer to the extremum). Negative curvature indicates an orbit that is a local maximum, whereas positive curvature indicates a local minimum.

--The standard deviations that are listed in the output as estimates of the error bounds of the final averaged quantities (see section 2.6 of the Rourke & Julian CPC 2012 article) are sample standard deviations, since the multiple copies of a given orbit found throughout the super cell are essentially repeated numerical samples of the true orbit. As such, if only one copy of an orbit is found, the standard deviation is undefined, and is therefore listed as NaN in the output. On the other hand, the coordinate standard deviations of the points on the orbit outlines (see sections 2.4 and 2.5 of the Rourke & Julian CPC 2012 article) are correctly expressed as population standard deviations, since they are used as estimates of overall orbit sizes, rather than "samples" of a single value.


7. CONTACT

Please contact the author (Patrick Rourke, at rourke@gmail.com) with any questions or comments.

