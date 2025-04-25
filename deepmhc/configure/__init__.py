import os
Config_Dir = os.path.dirname(__file__)

MHCII_Data_Conf = os.path.join(os.path.dirname(__file__), "mhcii_data.yaml")
ModelII_EL_Conf = os.path.join(os.path.dirname(__file__), "deepmhcii.yaml")
ModelII_ELContext_Conf = os.path.join(os.path.dirname(__file__), "deepmhcii_context.yaml")

MHCI_Data_Conf = os.path.join(os.path.dirname(__file__), "mhci_data.yaml")
ModelI_Neo_Conf = os.path.join(os.path.dirname(__file__), "neomhci.yaml")
ModelI_EL_Conf = os.path.join(os.path.dirname(__file__), "deepmhci.yaml")