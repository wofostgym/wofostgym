"""Handles Respiration of crop 

Written by: Anonymous (allard.dewit@wur.nl), April 2014
Modified by Anonymous Authors, 2024
"""
from datetime import date

from ..utils.traitlets import Float
from ..base import ParamTemplate, SimulationObject, RatesTemplate, VariableKiosk
from ..util import AfgenTrait
from ..nasapower import WeatherDataProvider

class WOFOST_Maintenance_Respiration(SimulationObject):
    """Maintenance respiration in WOFOST
    
    WOFOST calculates the maintenance respiration as proportional to the dry
    weights of the plant organs to be maintained, where each plant organ can be
    assigned a different maintenance coefficient. Multiplying organ weight
    with the maintenance coeffients yields the relative maintenance respiration
    (`RMRES`) which is than corrected for senescence (parameter `RFSETB`). Finally,
    the actual maintenance respiration rate is calculated using the daily mean
    temperature, assuming a relative increase for each 10 degrees increase
    in temperature as defined by `Q10`.

    **Simulation parameters:** (To be provided in cropdata dictionary):
    
    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    Q10      Relative increase in maintenance repiration    SCr       -
             rate with each 10 degrees increase in
             temperature
    RMR      Relative maintenance respiration rate for
             roots                                          SCr     |kg CH2O kg-1 d-1|
    RMS      Relative maintenance respiration rate for
             stems                                          SCr     |kg CH2O kg-1 d-1|
    RML      Relative maintenance respiration rate for
             leaves                                         SCr     |kg CH2O kg-1 d-1|
    RMO      Relative maintenance respiration rate for
             storage organs                                 SCr     |kg CH2O kg-1 d-1|
    RFSETB   Reduction factor  for senescence as            SCr       - 
             function of DVS
    =======  ============================================= =======  ============
    

    **State and rate variables:**
    
    `WOFOSTMaintenanceRespiration` returns the potential maintenance respiration PMRES
     directly from the `__call__()` method, but also includes it as a rate variable
     within the object.

     **Rate variables:**

    =======  ================================================ ==== =============
     Name     Description                                      Pbl      Unit
    =======  ================================================ ==== =============
    PMRES    Potential maintenance respiration rate             N  |kg CH2O ha-1 d-1|
    =======  ================================================ ==== =============

    **Signals send or handled**
    
    None
    
    **External dependencies:**
    
    =======  =================================== =============================  ============
     Name     Description                         Provided by                    Unit
    =======  =================================== =============================  ============
    DVS      Crop development stage              DVS_Phenology                  -
    WRT      Dry weight of living roots          WOFOST_Root_Dynamics           |kg ha-1|
    WST      Dry weight of living stems          WOFOST_Stem_Dynamics           |kg ha-1|
    WLV      Dry weight of living leaves         WOFOST_Leaf_Dynamics           |kg ha-1|
    WSO      Dry weight of living storage organs WOFOST_Storage_Organ_Dynamics  |kg ha-1|
    =======  =================================== =============================  ============


    """
    
    class Parameters(ParamTemplate):
        Q10 = Float(-99.)
        RMR = Float(-99.)
        RML = Float(-99.)
        RMS = Float(-99.)
        RMO = Float(-99.)
        RFSETB = AfgenTrait()

    class RateVariables(RatesTemplate):
        PMRES = Float(-99.)

    def initialize(self, day:date, kiosk:VariableKiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish="PMRES")
        self.kiosk = kiosk
        
    def __call__(self, day:date, drv:WeatherDataProvider):
        """Calculate the maintenence respiration of the crop
        """
        p = self.params
        kk = self.kiosk
        
        RMRES = (p.RMR * kk["WRT"] +
                 p.RML * kk["WLV"] +
                 p.RMS * kk["WST"] +
                 p.RMO * kk["WSO"])
        RMRES *= p.RFSETB(kk["DVS"])
        TEFF = p.Q10**((drv.TEMP-25.)/10.)
        self.rates.PMRES = RMRES * TEFF
        return self.rates.PMRES
    
    def reset(self):
        """Reset states and rates
        """
        self.rates.PMRES = 0

class Perennial_WOFOST_Maintenance_Respiration(SimulationObject):
    """Maintenance respiration in WOFOST
    
    WOFOST calculates the maintenance respiration as proportional to the dry
    weights of the plant organs to be maintained, where each plant organ can be
    assigned a different maintenance coefficient. Multiplying organ weight
    with the maintenance coeffients yields the relative maintenance respiration
    (`RMRES`) which is than corrected for senescence (parameter `RFSETB`). Finally,
    the actual maintenance respiration rate is calculated using the daily mean
    temperature, assuming a relative increase for each 10 degrees increase
    in temperature as defined by `Q10`.

    **Simulation parameters:** (To be provided in cropdata dictionary):
    
    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    Q10      Relative increase in maintenance repiration    SCr       -
             rate with each 10 degrees increase in
             temperature
    RMR      Relative maintenance respiration rate for
             roots                                          SCr     |kg CH2O kg-1 d-1|
    RMS      Relative maintenance respiration rate for
             stems                                          SCr     |kg CH2O kg-1 d-1|
    RML      Relative maintenance respiration rate for
             leaves                                         SCr     |kg CH2O kg-1 d-1|
    RMO      Relative maintenance respiration rate for
             storage organs                                 SCr     |kg CH2O kg-1 d-1|
    RFSETB   Reduction factor  for senescence as            SCr       - 
             function of DVS
    =======  ============================================= =======  ============
    

    **State and rate variables:**
    
    `WOFOSTMaintenanceRespiration` returns the potential maintenance respiration PMRES
     directly from the `__call__()` method, but also includes it as a rate variable
     within the object.

     **Rate variables:**

    =======  ================================================ ==== =============
     Name     Description                                      Pbl      Unit
    =======  ================================================ ==== =============
    PMRES    Potential maintenance respiration rate             N  |kg CH2O ha-1 d-1|
    =======  ================================================ ==== =============

    **Signals send or handled**
    
    None
    
    **External dependencies:**
    
    =======  =================================== =============================  ============
     Name     Description                         Provided by                    Unit
    =======  =================================== =============================  ============
    DVS      Crop development stage              DVS_Phenology                  -
    WRT      Dry weight of living roots          WOFOST_Root_Dynamics           |kg ha-1|
    WST      Dry weight of living stems          WOFOST_Stem_Dynamics           |kg ha-1|
    WLV      Dry weight of living leaves         WOFOST_Leaf_Dynamics           |kg ha-1|
    WSO      Dry weight of living storage organs WOFOST_Storage_Organ_Dynamics  |kg ha-1|
    =======  =================================== =============================  ============


    """
    
    class Parameters(ParamTemplate):
        Q10 = Float(-99.)
        RMR = AfgenTrait()
        RML = AfgenTrait()
        RMS = AfgenTrait()
        RMO = AfgenTrait()
        RFSETB = AfgenTrait()

    class RateVariables(RatesTemplate):
        PMRES = Float(-99.)

    def initialize(self, day:date, kiosk:VariableKiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish="PMRES")
        self.kiosk = kiosk
        
    def __call__(self, day:date, drv:WeatherDataProvider):
        """Calculate the maintenence respiration of the crop
        """
        p = self.params
        kk = self.kiosk
        
        RMRES = (p.RMR(kk.AGE) * kk["WRT"] +
                 p.RML(kk.AGE) * kk["WLV"] +
                 p.RMS(kk.AGE) * kk["WST"] +
                 p.RMO(kk.AGE) * kk["WSO"])
        RMRES *= p.RFSETB(kk["DVS"])
        TEFF = p.Q10**((drv.TEMP-25.)/10.)
        self.rates.PMRES = RMRES * TEFF
        return self.rates.PMRES
    
    def reset(self):
        """Reset states and rates
        """
        self.rates.PMRES = 0
