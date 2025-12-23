#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

# Imports from Pyomo
from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    Objective,
    value,
    assert_optimal_termination,
    TransformationFactory,
    units as pyunits,
)
from pyomo.network import Arc

# Imports from IDAES
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.models.unit_models import Feed, Product
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.initialization import propagate_state

# Imports from WaterTAP
from watertap.costing import WaterTAPCosting
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump
from watertap.unit_models.pressure_changer import EnergyRecoveryDevice
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.core.solvers import get_solver



def main():
    m = build()
    scale_system(m)
    add_costing(m)
    initialize_system(m)
    results = solve_system(m)
    display_costing(m)

    return m, results


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Add seawater property models
    m.fs.properties = SeawaterParameterBlock()

    # Add feed
    m.fs.feed = Feed(property_package=m.fs.properties)
    # Set feed stream
    m.fs.feed.properties[0].flow_vol_phase["Liq"].fix(1e-3)
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "TDS"].fix(35)
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)

    # Add pump
    m.fs.pump = Pump(property_package=m.fs.properties)
    # Set pump parameters
    m.fs.pump.efficiency_pump.fix(0.80)
    m.fs.pump.control_volume.properties_out[0].pressure.fix(75 * pyunits.bar)

    # Add 0D reverse osmosis unit
    m.fs.RO = ReverseOsmosis0D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )

    # Set RO parameters
    m.fs.RO.A_comp.fix(4.2e-12)
    m.fs.RO.B_comp.fix(3.5e-8)
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(0.5)
    m.fs.RO.feed_side.channel_height.fix(1e-3)
    m.fs.RO.feed_side.spacer_porosity.fix(0.97)
    m.fs.RO.permeate.pressure[0].fix(101325)
    m.fs.RO.area.fix(50)

    # Add Energy Recovery Device
    m.fs.erd = EnergyRecoveryDevice(property_package=m.fs.properties)
    # Set ERD parameters
    m.fs.erd.efficiency_pump.fix(0.95)
    m.fs.erd.outlet.pressure[0].fix(101325)

    # Add product stream
    m.fs.product = Product(property_package=m.fs.properties)

    # Define the connectivity using Arcs
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_RO = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.RO_to_erd = Arc(source=m.fs.RO.retentate, destination=m.fs.erd.inlet)
    m.fs.RO_to_product = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)

    # Use the TransformationFactory to expand the arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def scale_system(m):
    # Set Scaling factors
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
    )
    set_scaling_factor(m.fs.pump.control_volume.work, 1e-3)
    set_scaling_factor(m.fs.erd.control_volume.work, 1e-3)
    set_scaling_factor(m.fs.RO.area, 1e-2)
    calculate_scaling_factors(m)

    print(f"dof = {degrees_of_freedom(m)}")


def add_costing(m):
    # Add costing
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2020

    # Add costing blocks to unit models
    m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.erd.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    # Process costing
    m.fs.costing.cost_process()

    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol_phase["Liq"])
    m.fs.costing.add_specific_energy_consumption(
        m.fs.product.properties[0].flow_vol_phase["Liq"], name="SEC"
    )


def initialize_system(m):
    solver = get_solver()
    # Solve feed
    solver.solve(m.fs.feed)

    # Propagate state from feed to pump
    propagate_state(m.fs.feed_to_pump)
    # Initialize pump
    m.fs.pump.initialize()
    # Propagate state from pump to RO
    propagate_state(m.fs.pump_to_RO)
    # Initialize RO
    m.fs.RO.initialize()
    # Propagate state from RO to ERD and product
    propagate_state(m.fs.RO_to_erd)
    propagate_state(m.fs.RO_to_product)
    # Initialize the product and ERD
    m.fs.product.initialize()
    m.fs.erd.initialize()


def solve_system(m):
    assert degrees_of_freedom(m) == 0
    solver = get_solver()
    results = solver.solve(m)
    assert_optimal_termination(results)

    return results


def display_costing(m):
    m.fs.costing.LCOW.display()
    m.fs.costing.SEC.display()
    m.fs.costing.aggregate_flow_costs.display()
    m.fs.costing.aggregate_flow_electricity.display()
    m.fs.costing.SEC_component.display()

def sensitivity_analysis_on_recovery(m, recoveries):
    #TODO: Implement sensitivity analysis on recovery for upfront demonstration purposes before diving into parameter sweep setup
    pass

if __name__ == "__main__":
    m, results = main()
