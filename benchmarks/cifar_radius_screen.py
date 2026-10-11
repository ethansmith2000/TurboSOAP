"""Attach read-only radius probes to fixed-placement Frobenius control training."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]


def main():
    import cifar_strict_transport as transport
    from radius_probe import RadiusProbe
    import cifar_affinity_screen
    transport.StrictTransportProbe=RadiusProbe
    cifar_affinity_screen.main()


if __name__=='__main__':main()
