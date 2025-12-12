"""
Script to vary bar angle in SynthPop bulge model and generate star catalogs
for specific galactic coordinates (l, b) with given solid angle cone.

Based on the bulge.popjson configuration from Huston2025 model.
"""

import synthpop
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class BarAngleVariationTool:    
    def __init__(self, base_config_file: str = 'huston2025_defaults.synthpop_conf'):
        self.base_config_file = base_config_file
        self.temp_dir = None
        self.models = {}
        
    def setup_model(self, bar_angle: float, output_name: Optional[str] = None) -> synthpop.SynthPop:
        """
        Set up SynthPop model with bar angle passed directly
        This method tries to override just the population_density_kwargs parameter
        
        Args:
            bar_angle: Bar angle in degrees
            output_name: Optional custom output name
            
        Returns:
            Configured SynthPop model
        """
        if output_name is None:
            output_name = f'bar_angle_{bar_angle:.1f}'
        
        # Try to override just the population density parameters for bulge
        # This approach tries to pass the bar angle directly without creating a config file
        model = synthpop.SynthPop(
            self.base_config_file,
            extinction_map_kwargs={
                'name': 'maps_from_dustmaps', 
                'dustmap_name': 'marshall'
            },
            chosen_bands=['Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I', "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"],
            maglim=['Gaia_G_EDR3', 20, "keep"],
            post_processing_kwargs=[{"name": "ProcessDarkCompactObjects", "remove": False}],
            name_for_output=output_name,
            # Try to override just the population density parameters
            population_density_kwargs={
                'bulge': {
                    "name": "triaxial_bulge",
                    "triaxial_type": "E3", 
                    "density_unit": "mass",
                    "x0": 0.67,
                    "y0": 0.29,
                    "z0": 0.27,
                    "rho0": 1.32585e10,
                    "bar_angle": bar_angle  # This is what we're varying
                }
            }
        )
        
        # Initialize populations
        model.init_populations()
        
        # Cache the model
        self.models[bar_angle] = model
        
        return model

    def generate_catalog(self, 
                        bar_angle: float, 
                        l_deg: float, 
                        b_deg: float, 
                        solid_angle: float,
                        solid_angle_unit: str = 'deg^2') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate star catalog for given parameters
        
        Args:
            bar_angle: Bar angle in degrees
            l_deg: Galactic longitude in degrees
            b_deg: Galactic latitude in degrees
            solid_angle: Solid angle of the cone
            solid_angle_unit: Unit of solid angle ('deg^2' or 'sr')
            
        Returns:
            Tuple of (catalog DataFrame, distance distribution)
        """
        # Get or create model for this bar angle
        model = self.setup_model(bar_angle)
        
        # Generate catalog
        catalog, distance_dist = model.process_location(
            l_deg=l_deg, 
            b_deg=b_deg, 
            solid_angle=solid_angle,
            solid_angle_unit=solid_angle_unit
        )
        
        return catalog, distance_dist