import argparse

def args_parser():
    """Function to take input command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fermi Surface Plotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        metavar="\b",
        type=str,
        help="Name of the bxsf file in which data is stored",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--interpolation-factor",
        metavar="\b",
        type=int,
        default=1,
        help="Degree of interpolation of Fermi surface",
    )
    parser.add_argument(
        "-b",
        "--bands",
        metavar="\b",
        type=str,
        help="List of bands to include in plot",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        metavar="\b",
        type=float,
        help="Zoom factor of saved image",
        default=1.0,
    )
    parser.add_argument(
        "-se",
        "--shift-energy",
        metavar="\b",
        type=float,
        help="Shift in the Fermi energy from the DFT/input value",
        default=0.0,
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        metavar="\b",
        type=float,
        help="Azimuthal angle of camera",
        default=65,
    )
    parser.add_argument(
        "-e",
        "--elevation",
        metavar="\b",
        type=float,
        help="Elevation angle of camera",
        default=30,
    )
    parser.add_argument(
        "-o",
        "--order",
        metavar="\b",
        type=int,
        help="Order of spline for interpolation between 1 and 5",
        default=3,
    )
    parser.add_argument(
        "-s",
        "--spin",
        metavar="\b",
        type=str,
        choices=["u", "d", "ud", "n"],
        help="Whether the bands are spin polarised and hence which spins to choose. Options: ['u', 'd', 'ud', 'n']",
        default="n",
    )
    parser.add_argument(
        "-vf",
        "--fermi-velocity",
        metavar="\b",
        type=bool,
        help="Compute the Fermi velocity on the surface",
        default=False,
    )
    parser.add_argument(
        "-lw",
        "--line-width",
        metavar="\b",
        type=float,
        help="Linewidth for the Brillouin Zone plot",
        default=2.0,
    )
    parser.add_argument(
        "-op",
        "--opacity",
        metavar="\b",
        type=float,
        help="Opacity of the Fermi surface plot, Range: [0, 1]",
        default=1.0,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        metavar="\b",
        type=float,
        help="Resoltuion of saved pdf image",
        default=1.0,
    )
    parser.add_argument(
        "-int",
        "--interactive",
        metavar="\b",
        type=bool,
        help="Interactive Fermi Surface visualisation",
        default=True,
    )
    parser.add_argument(
        "-bn",
        "--band-name",
        metavar="\b",
        type=bool,
        help="Add Band index to the plots",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        metavar="\b",
        type=str,
        help="The format of the saved figure (png, jpg, pdf). Multiple choice are allowed.",
        choices=["png", "jpg", "pdf"],
        default="pdf",
    )
    parser.add_argument(
        "-pr",
        "--projection",
        metavar="\b",
        type=str,
        help="Whether surfaces are projected in parallel mode or perspective",
        choices=["parallel", "perspective", "par", "per"],
        default="parallel",
    )
    parser.add_argument(
        "-sc",
        "--scalar",
        metavar="\b",
        type=str,
        help="Name of file containing a scalar field to plot on the Fermi surface",
        default="None",
    )
    parser.add_argument(
        "-cmp",
        "--colourmap",
        metavar="\b",
        type=str,
        help="The plot colourmap",
        default="turbo",
    )
    parser.add_argument(
        "-cb",
        "--colourbar",
        metavar="\b",
        type=bool,
        help="Whether to plot colorbar for scalar fields or vf",
        default=False,
    )
    parser.add_argument(
        "-norm",
        "--normalise",
        metavar="\b",
        type=bool,
        help="Whether to renormalise the plot around 0",
        default=False,
    )
    parser.add_argument(
        "-pow",
        "--power",
        metavar="\b",
        type=int,
        help="Power scaling of scalar field",
        default=1,
    )
    parser.add_argument(
        "-nc",
        "--no-clip",
        metavar="\b",
        type=bool,
        help="Whether to clip isosurface",
        default=False,
    )
    parser.add_argument(
        "-sk",
        "--skeaf",
        metavar="\b",
        type=bool,
        help="Whether to run SKEAF to calculate extremal orbital areas",
        default=False,
    )
    parser.add_argument(
        "-sk-t",
        "--skeaf-theta",
        metavar="\b",
        type=float,
        help="Theta angle for extremal loop calculation",
        default=0.0,
    )
    parser.add_argument(
        "-sk-p",
        "--skeaf-phi",
        metavar="\b",
        type=float,
        help="Phi angle for extremal loop calculation",
        default=0.0,
    )
    parser.add_argument(
        "-sk-min",
        "--skeaf-min",
        metavar="\b",
        type=float,
        help="Minimum extremal FS frequency (kT)",
        default=0.0,
    )
    parser.add_argument(
        "-sk-mfd",
        "--skeaf-min-frac-diff",
        metavar="\b",
        type=float,
        help="Maximum fractional diff. between orbit freqs. for averaging",
        default=0.01,
    )
    parser.add_argument(
        "-sk-md",
        "--skeaf-min-dist",
        metavar="\b",
        type=float,
        help="Maximum distance between orbit avg. coords. for averaging",
        default=0.05,
    )
    parser.add_argument(
        "-sk-i",
        "--skeaf-interpolate",
        metavar="\b",
        type=int,
        help="Interpolated number of points per single side",
        default=100,
    )
    parser.add_argument(
        "-sk-li",
        "--skeaf-line-interpolate",
        metavar="\b",
        type=int,
        help="Interpolation factor for SKEAF orbits lines when plotting",
        default=1,
    )
    parser.add_argument(
        "-sk-lo",
        "--skeaf-line-order",
        metavar="\b",
        type=int,
        help="Interpolation order for SKEAF orbits lines when plotting",
        default=3,
    )
    parser.add_argument(
        "-sk-ls",
        "--skeaf-line-smoothness",
        metavar="\b",
        type=float,
        help="Spline smoothness for SKEAF orbits lines when plotting, should be very small < 0.0001",
        default=0.0,
    )
    parser.add_argument(
        "-sk-lw",
        "--skeaf-linewidth",
        metavar="\b",
        type=float,
        help="Linewidth of SKEAF orbit in plot",
        default=1.5,
    )
    parser.add_argument(
        "-sk-c",
        "--skeaf-colour",
        metavar="\b",
        type=str,
        help="Colour of SKEAF orbit plot",
        default="red",
    )
    parser.add_argument(
        "-clipx",
        "--clip-in-x",
        metavar="\b",
        type=float,
        help="Clip by given amount in the x-direction",
        default=0,
    )
    parser.add_argument(
        "-clipy",
        "--clip-in-y",
        metavar="\b",
        type=float,
        help="Clip by given amount in the y-direction",
        default=0,
    )
    parser.add_argument(
        "-clipz",
        "--clip-in-z",
        metavar="\b",
        type=float,
        help="Clip by given amount in the z-direction",
        default=0,
    )


    return parser

