FC=ifort
opt=-mcmodel=medium -shared-intel -xHost -O3 -ipo -qopenmp

skeafmake: 
	$(FC) $(opt) skeaf_v1p3p0_r149.F90 -o skeaf
