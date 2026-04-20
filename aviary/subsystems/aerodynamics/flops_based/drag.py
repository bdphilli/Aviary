import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.variable_info.variables import Aircraft, Dynamic


class ScaledCD(om.ExplicitComponent):
    """
    Apply the final drag coefficient factors to the unscaled drag.

    These optional factors (default: 1.0) increase or decrease the drag
    coefficient before calculating drag.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, units='unitless')

        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')

        self.add_input(
            'CD_prescaled', val=np.ones(nn), units='unitless', desc='total drag coefficient'
        )

        add_aviary_output(self, Dynamic.Vehicle.DRAG_COEFFICIENT, shape=nn, units='unitless')

    def setup_partials(self):
        self.declare_partials(
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            [
                'CD_prescaled',
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
            ],
        )

        self.declare_partials(
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            Dynamic.Atmosphere.MACH,
            dependent=False,
        )

        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            'CD_prescaled',
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Atmosphere.MACH]

        CD_prescaled = inputs['CD_prescaled']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD_prescaled * FCDSUB
        CD_scaled[idx_sup] = CD_prescaled[idx_sup] * FCDSUP
        outputs[Dynamic.Vehicle.DRAG_COEFFICIENT] = CD_scaled

    def compute_partials(self, inputs, partials):
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Atmosphere.MACH]
        CD_prescaled = inputs['CD_prescaled']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD_prescaled * FCDSUB
        CD_scaled[idx_sup] = CD_prescaled[idx_sup] * FCDSUP

        idx_sub = np.where(M < 1.0)
        dCD = np.ones_like(CD_prescaled)
        dCD[idx_sub] = FCDSUB
        dCD[idx_sup] = FCDSUP
        partials[Dynamic.Vehicle.DRAG_COEFFICIENT, 'CD_prescaled'] = dCD

        dF = np.zeros_like(CD_prescaled)
        dF[idx_sub] = CD_prescaled[idx_sub]
        partials[Dynamic.Vehicle.DRAG_COEFFICIENT, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR] = dF

        dF = np.zeros_like(CD_prescaled)
        dF[idx_sup] = CD_prescaled[idx_sup]
        partials[Dynamic.Vehicle.DRAG_COEFFICIENT, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR] = (
            dF
        )


class SimpleDrag(om.ExplicitComponent):
    """Calculate drag as a function of wing area, dynamic pressure, and drag coefficient."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')
        add_aviary_input(self, Dynamic.Vehicle.DRAG_COEFFICIENT, shape=nn, units='unitless')

        add_aviary_output(self, Dynamic.Vehicle.DRAG, shape=nn, units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(Dynamic.Vehicle.DRAG, Aircraft.Wing.AREA)

        self.declare_partials(
            Dynamic.Vehicle.DRAG,
            [Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Vehicle.DRAG_COEFFICIENT],
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CD = inputs[Dynamic.Vehicle.DRAG_COEFFICIENT]

        outputs[Dynamic.Vehicle.DRAG] = q * S * CD

    def compute_partials(self, inputs, partials):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CD = inputs[Dynamic.Vehicle.DRAG_COEFFICIENT]

        partials[Dynamic.Vehicle.DRAG, Aircraft.Wing.AREA] = q * CD
        partials[Dynamic.Vehicle.DRAG, Dynamic.Atmosphere.DYNAMIC_PRESSURE] = S * CD
        partials[Dynamic.Vehicle.DRAG, Dynamic.Vehicle.DRAG_COEFFICIENT] = q * S


class TotalDrag(om.Group):
    """
    Calculate drag as a function of wing area, dynamic pressure, and lift-dependent and
    lift-independent drag coefficients.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    dependent drag coefficient before calculating the total drag coefficient.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    independent drag coefficient before calculating the total drag coefficient.

    Note, the lift-dependent drag coefficient includes contributions from the pressure
    drag coefficient.

    Apply optional factors (default: 1.0) for increasing or decreasing the total drag
    coefficient before calculating drag. The effect is cumulative with the above
    optional factors.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        FCDI_desc = CoreMetaData[Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR]['desc']
        FCD0_desc = CoreMetaData[Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR]['desc']

        CDF_SCALER_desc = CoreMetaData[Dynamic.Vehicle.DRAG_POLAR_CDF_SCALER]['desc']
        CDC_SCALER_desc = CoreMetaData[Dynamic.Vehicle.DRAG_POLAR_CDC_SCALER]['desc']
        CDP_SCALER_desc = CoreMetaData[Dynamic.Vehicle.DRAG_POLAR_CDP_SCALER]['desc']
        CDI_SCALER_desc = CoreMetaData[Dynamic.Vehicle.DRAG_POLAR_CDI_SCALER]['desc']
        CD_RESIDUAL_desc = CoreMetaData[Dynamic.Vehicle.DRAG_POLAR_RESIDUAL]['desc']

        def vec(desc):
            # fresh array per input; a shared default array across ExecComp
            # inputs would alias state between components
            return dict(val=np.ones(nn), units='unitless', desc=desc)

        # The drag buildup is split into three small ExecComps (mirroring the
        # stock TotalDrag topology) rather than one large expression. Keeping
        # each component at <= 5 inputs preserves the per-node diagonal
        # declared-sparsity pattern of the stock implementation; a single
        # wide ExecComp colors more conservatively and inflates the declared
        # (but numerically zero) entries in the optimizer's constraint
        # jacobian.
        #
        # The classical-induced-drag input is named `CDI_IND` (not plain
        # `CDI`) to avoid conflicting with the legacy lumped `CDI` output
        # that `ComputedDrag` still emits for backward compatibility.
        cd0_comp = self.add_subsystem(
            'scaled_zero_lift_drag',
            om.ExecComp(
                'CD0_scaled = CDF * CDF_SCALER + CDC * CDC_SCALER',
                CDF=vec('skin-friction drag coefficient'),
                CDC=vec('compressibility drag coefficient'),
                CDF_SCALER=vec(CDF_SCALER_desc),
                CDC_SCALER=vec(CDC_SCALER_desc),
                CD0_scaled=vec('scaled lift-independent drag coefficient'),
            ),
            promotes_inputs=[
                'CDF',
                'CDC',
                ('CDF_SCALER', Dynamic.Vehicle.DRAG_POLAR_CDF_SCALER),
                ('CDC_SCALER', Dynamic.Vehicle.DRAG_POLAR_CDC_SCALER),
            ],
            promotes_outputs=['CD0_scaled'],
        )
        cd0_comp.declare_coloring(show_summary=False)

        cdi_comp = self.add_subsystem(
            'scaled_lift_dependent_drag',
            om.ExecComp(
                'CDI_scaled = CDP * CDP_SCALER + CDI_IND * CDI_SCALER',
                CDP=vec('lift-dependent pressure (wave) drag coefficient'),
                CDI_IND=vec('classical induced drag coefficient (from vortex lift)'),
                CDP_SCALER=vec(CDP_SCALER_desc),
                CDI_SCALER=vec(CDI_SCALER_desc),
                CDI_scaled=vec('scaled lift-dependent drag coefficient'),
            ),
            promotes_inputs=[
                'CDP',
                'CDI_IND',
                ('CDP_SCALER', Dynamic.Vehicle.DRAG_POLAR_CDP_SCALER),
                ('CDI_SCALER', Dynamic.Vehicle.DRAG_POLAR_CDI_SCALER),
            ],
            promotes_outputs=['CDI_scaled'],
        )
        cdi_comp.declare_coloring(show_summary=False)

        total_drag_comp = self.add_subsystem(
            'total_drag_coeff',
            om.ExecComp(
                'CD_prescaled = CDI_scaled * FCDI + CD0_scaled * FCD0 + CD_RESIDUAL',
                CDI_scaled=vec('scaled lift-dependent drag coefficient'),
                CD0_scaled=vec('scaled lift-independent drag coefficient'),
                FCDI=dict(val=1.0, units='unitless', desc=FCDI_desc),
                FCD0=dict(val=1.0, units='unitless', desc=FCD0_desc),
                # Additive residual applied on top of the whole (scaled) FLOPS
                # prediction. Default 0.0 preserves stock output.
                CD_RESIDUAL=dict(val=np.zeros(nn), units='unitless', desc=CD_RESIDUAL_desc),
                CD_prescaled=vec('total drag coefficient'),
            ),
            promotes_inputs=[
                'CDI_scaled',
                'CD0_scaled',
                ('FCDI', Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR),
                ('FCD0', Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR),
                ('CD_RESIDUAL', Dynamic.Vehicle.DRAG_POLAR_RESIDUAL),
            ],
            promotes_outputs=['*'],
        )
        total_drag_comp.declare_coloring(show_summary=False)

        self.add_subsystem('simple_CD', ScaledCD(num_nodes=nn), promotes=['*'])
        self.add_subsystem('simple_drag', SimpleDrag(num_nodes=nn), promotes=['*'])
