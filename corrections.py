from __future__ import annotations

from dataclasses import dataclass
from math import cos, cosh, exp, isfinite, pi, sqrt

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SI konstansok: ezeket csak a vegso, merheto sebessegi egyutthato
# cm^3 molecule^-1 s^-1 egysegehez, illetve egyszeru konverziokhoz hasznaljuk.
# Az Eckart-alagutkorrekcio maga lentebb atomi egysegekben fut.
# ---------------------------------------------------------------------------
KB_SI = 1.380649e-23  # J K^-1
H_SI = 6.62607015e-34  # J s
C_CM_S = 2.99792458e10  # cm s^-1
EH_J = 4.3597447222060e-18  # J
AMU_KG = 1.66053906660e-27  # kg
NA = 6.02214076e23  # mol^-1


# ---------------------------------------------------------------------------
# Atomi egysegek az Eckart-reszhez.
#
# Energiak: Hartree (Eh)
# Tomegek: elektron tomeg (me)
# Hosszak: Bohr (a0)
# Frekvencia: 1 / atomido, azaz a cm^-1 frekvenciat eloszor Hz-be,
# majd atomic frequency-be valtjuk.
# ---------------------------------------------------------------------------
KB_AU = 3.167e-6  # Eh K^-1
TIME_AU_S = 2.4188843265857e-17  # s
AMU_TO_ME = 1822.888486209  # 1 amu hany elektron-tomeg
H_AU = 2.0 * pi  # Atomi egysegben hbar = 1, ezert h = 2*pi

# Egy cm^-1 rezgesi kvantum energiaja Hartree-ban.
HC_PER_CM_HARTREE = H_SI * C_CM_S / EH_J


@dataclass(frozen=True)
class Species:
    name: str
    energy_hartree: float
    mass_amu: float
    freqs_cm: tuple[float, ...]
    rot_constants_cm: tuple[float, ...] = ()
    symmetry_number: int = 1
    electronic_degeneracy: int = 1
    zpe_hartree: float | None = None

    def positive_freqs(self, cutoff_cm: float = 0.0) -> tuple[float, ...]:
        """Csak a stabil rezgesi modokat tartjuk meg."""
        return tuple(nu for nu in self.freqs_cm if nu > cutoff_cm)

    def zpe_eh(self, freq_cutoff_cm: float = 0.0) -> float:
        """Nullponti energia Hartree-ban."""
        if self.zpe_hartree is not None:
            return self.zpe_hartree
        return 0.5 * HC_PER_CM_HARTREE * sum(
            self.positive_freqs(freq_cutoff_cm)
        )

    def q_vib(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        """Harmonikus rezgesi allapotosszeg."""
        q = 1.0
        kt_eh = KB_AU * temperature_k

        for nu_cm in self.positive_freqs(freq_cutoff_cm):
            x = HC_PER_CM_HARTREE * nu_cm / kt_eh
            q *= 1.0 / (1.0 - exp(-x))

        return q

    def q_rot(self, temperature_k: float) -> float:
        """Merev rotor allapotosszeg."""
        if len(self.rot_constants_cm) == 0:
            return 1.0

        if len(self.rot_constants_cm) == 1:
            theta = HC_PER_CM_HARTREE * self.rot_constants_cm[0] / KB_AU
            return temperature_k / (self.symmetry_number * theta)

        if len(self.rot_constants_cm) == 3:
            theta_a, theta_b, theta_c = (
                HC_PER_CM_HARTREE * b_cm / KB_AU
                for b_cm in self.rot_constants_cm
            )
            return (
                sqrt(pi)
                * temperature_k**1.5
                / (self.symmetry_number * sqrt(theta_a * theta_b * theta_c))
            )

        raise ValueError(
            f"{self.name}: 0, 1 vagy 3 rotacios allandot adj meg cm^-1-ben."
        )

    def q_int(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        """Belső allapotosszeg: elektronikus * rotacios * rezgesi."""
        return (
            self.electronic_degeneracy
            * self.q_rot(temperature_k)
            * self.q_vib(temperature_k, freq_cutoff_cm)
        )


def reduced_mass_kg(a: Species, b: Species) -> float:
    """Redukalt tomeg kg-ban a klasszikus TST transzlacios tagjahoz."""
    mu_amu = a.mass_amu * b.mass_amu / (a.mass_amu + b.mass_amu)
    return mu_amu * AMU_KG


def zpe_corrected_barrier_eh(
    reactant_a: Species,
    reactant_b: Species,
    transition_state: Species,
    freq_cutoff_cm: float = 0.0,
) -> float:
    """ZPE-vel korrigalt gatmagassag Hartree-ban."""
    delta_e_elec = (
        transition_state.energy_hartree
        - reactant_a.energy_hartree
        - reactant_b.energy_hartree
    )
    delta_zpe = (
        transition_state.zpe_eh(freq_cutoff_cm)
        - reactant_a.zpe_eh(freq_cutoff_cm)
        - reactant_b.zpe_eh(freq_cutoff_cm)
    )
    return delta_e_elec + delta_zpe


def wavenumber_cm_to_hz(nu_cm: float) -> float:
    """cm^-1 -> s^-1."""
    return abs(nu_cm) * C_CM_S


def hz_to_atomic_frequency(nu_hz: float) -> float:
    """s^-1 -> atomido^-1."""
    return nu_hz * TIME_AU_S


def calculate_wigner_kappa(
    imag_freq_cm: float,
    temperature_k: float,
) -> float:
    """Wigner-alagutkorrekcio a TS imaginarius frekvenciajabol.

    kappa_W = 1 + (h * |nu_imag| / (kB*T))^2 / 24
    """
    if temperature_k <= 0.0:
        raise ValueError("A homersekletnek pozitivnak kell lennie.")

    h_nu_eh = HC_PER_CM_HARTREE * abs(imag_freq_cm)
    x = h_nu_eh / (KB_AU * temperature_k)

    return 1.0 + x * x / 24.0


def amu_to_electron_mass(mass_amu: float) -> float:
    """amu -> elektron-tomeg."""
    return mass_amu * AMU_TO_ME


def eckart_width_bohr(
    delta_ef_eh: float,
    delta_er_eh: float,
    imag_freq_cm: float,
    eff_mass_me: float,
) -> float:
    """Eckart-gat karakterisztikus szelessege Bohrban."""
    if delta_ef_eh <= 0.0 or delta_er_eh <= 0.0:
        raise ValueError("Az Eckart-korrekciohoz mindket gatmagassag legyen pozitiv.")

    nu_hz = wavenumber_cm_to_hz(imag_freq_cm)
    nu_au = hz_to_atomic_frequency(nu_hz)

    reciprocal_sum = 1.0 / sqrt(delta_ef_eh) + 1.0 / sqrt(delta_er_eh)

    return (1.0 / nu_au) * sqrt(2.0 / eff_mass_me) / reciprocal_sum


def eckart_transmission_probability_au(
    energy_eh: float,
    a_eh: float,
    b_eh: float,
    width_bohr: float,
    eff_mass_me: float,
) -> float:
    """P(E) atomi egysegekben az aszimmetrikus Eckart-gatra."""
    if energy_eh <= 0.0:
        return 0.0

    alpha = sqrt(2.0 * eff_mass_me * energy_eh) * (width_bohr / (2.0 * pi))

    diff_e_a = energy_eh - a_eh
    if diff_e_a <= 0.0:
        return 0.0

    beta = sqrt(2.0 * eff_mass_me * diff_e_a) * (width_bohr / (2.0 * pi))

    radicand = (8.0 * eff_mass_me * b_eh * width_bohr**2) / (H_AU**2) - 1.0

    if radicand >= 0.0:
        delta = 0.5 * sqrt(radicand)
        cos_2pi_delta = cos(2.0 * pi * delta)
    else:
        gamma = 0.5 * sqrt(abs(radicand))
        cos_2pi_delta = cosh(2.0 * pi * gamma)

    try:
        numerator = cosh(2.0 * pi * (alpha - beta)) + cos_2pi_delta
        denominator = cosh(2.0 * pi * (alpha + beta)) + cos_2pi_delta
        p_e = 1.0 - numerator / denominator
    except OverflowError:
        p_e = 1.0 - exp(-4.0 * pi * beta)

    if not isfinite(p_e):
        return 0.0

    return max(0.0, min(1.0, float(p_e)))


def simpson_integral(func, lower: float, upper: float, intervals: int = 6000) -> float:
    """Egyszeru Simpson-integralas kulso csomagok nelkul."""
    if intervals % 2:
        intervals += 1

    step = (upper - lower) / intervals
    total = func(lower) + func(upper)

    for i in range(1, intervals):
        coefficient = 4.0 if i % 2 else 2.0
        total += coefficient * func(lower + i * step)

    return total * step / 3.0


def calculate_eckart_kappa_au(
    delta_ef_eh: float,
    delta_er_eh: float,
    imag_freq_cm: float,
    eff_mass_amu: float,
    temperature_k: float,
) -> float:
    """Eckart-alagut korrekcios tenyezo, kappa(T), atomi egysegekben."""
    eff_mass_me = amu_to_electron_mass(eff_mass_amu)

    a_eh = delta_ef_eh - delta_er_eh
    b_eh = (sqrt(delta_ef_eh) + sqrt(delta_er_eh)) ** 2

    width_bohr = eckart_width_bohr(
        delta_ef_eh=delta_ef_eh,
        delta_er_eh=delta_er_eh,
        imag_freq_cm=imag_freq_cm,
        eff_mass_me=eff_mass_me,
    )

    kt_eh = KB_AU * temperature_k

    def integrand(energy_eh: float) -> float:
        p_e = eckart_transmission_probability_au(
            energy_eh=energy_eh,
            a_eh=a_eh,
            b_eh=b_eh,
            width_bohr=width_bohr,
            eff_mass_me=eff_mass_me,
        )

        exponent = (delta_ef_eh - energy_eh) / kt_eh

        if exponent > 700.0:
            return 0.0

        return p_e * exp(exponent) / kt_eh

    upper_limit_eh = max(delta_ef_eh * 5.0, 20.0 * kt_eh)
    kappa = simpson_integral(integrand, 0.0, upper_limit_eh)

    return max(1.0, float(kappa))


def show_comparison_plot(
    temperatures: list[float],
    k_tst_list: list[float],
    k_wigner_list: list[float],
    k_eckart_list: list[float],
    kappa_wigner_list: list[float],
    kappa_eckart_list: list[float],
) -> None:
    """Az osszehasonlito grafikont kulon Matplotlib-ablakban jeleniti meg."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(temperatures, k_tst_list, "r--", label="Klasszikus TST")
    ax1.plot(temperatures, k_wigner_list, "g-.", label="Wigner-korrigalt TST")
    ax1.plot(temperatures, k_eckart_list, "b-", label="Eckart-korrigalt TST")
    ax1.set_yscale("log")
    ax1.set_xlabel("Homerseklet (T / K)")
    ax1.set_ylabel(r"$k(T)$  ($\mathrm{cm^3\ molecule^{-1}\ s^{-1}}$)")
    ax1.set_title("Sebessegi egyutthatok a homerseklet fuggvenyeben")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(temperatures, kappa_wigner_list, "g-.", label="Wigner")
    ax2.plot(temperatures, kappa_eckart_list, "b-", label="Eckart")
    ax2.set_yscale("log")
    ax2.set_xlabel("Homerseklet (T / K)")
    ax2.set_ylabel(r"Alaguttényezo ($\kappa$)")
    ax2.set_title("Wigner- es Eckart-korrekcio")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def canonical_tst_bimolecular_rate(
    reactant_a: Species,
    reactant_b: Species,
    transition_state: Species,
    product_a: Species,
    product_b: Species,
    temperature_k: float,
    imag_freq_cm: float,
    reaction_path_degeneracy: int = 1,
    freq_cutoff_cm: float = 0.0,
) -> dict[str, float]:
    """Kanonikus TST + Wigner- es Eckart-korrekcio."""
    delta_ef_eh = zpe_corrected_barrier_eh(
        reactant_a, reactant_b, transition_state, freq_cutoff_cm
    )
    delta_er_eh = zpe_corrected_barrier_eh(
        product_a, product_b, transition_state, freq_cutoff_cm
    )

    mu_kg = reduced_mass_kg(reactant_a, reactant_b)

    translational_factor_m3 = (
        H_SI * H_SI / (2.0 * pi * mu_kg * KB_SI * temperature_k)
    ) ** 1.5

    q_ratio = transition_state.q_int(temperature_k, freq_cutoff_cm) / (
        reactant_a.q_int(temperature_k, freq_cutoff_cm)
        * reactant_b.q_int(temperature_k, freq_cutoff_cm)
    )

    k_tst_m3 = (
        reaction_path_degeneracy
        * (KB_SI * temperature_k / H_SI)
        * translational_factor_m3
        * q_ratio
        * exp(-delta_ef_eh / (KB_AU * temperature_k))
    )

    kappa_eckart = calculate_eckart_kappa_au(
        delta_ef_eh=delta_ef_eh,
        delta_er_eh=delta_er_eh,
        imag_freq_cm=imag_freq_cm,
        eff_mass_amu=1.00782503223,
        temperature_k=temperature_k,
    )

    kappa_wigner = calculate_wigner_kappa(
        imag_freq_cm=imag_freq_cm,
        temperature_k=temperature_k,
    )

    k_eckart_m3 = k_tst_m3 * kappa_eckart
    k_wigner_m3 = k_tst_m3 * kappa_wigner

    return {
        "k_tst_cm3": k_tst_m3 * 1.0e6,
        "k_wigner_cm3": k_wigner_m3 * 1.0e6,
        "k_corrected_cm3": k_eckart_m3 * 1.0e6,
        "kappa_wigner": kappa_wigner,
        "kappa_eckart": kappa_eckart,
        "delta_Ef_kJ_mol": delta_ef_eh * EH_J * NA / 1000.0,
        "delta_Er_kJ_mol": delta_er_eh * EH_J * NA / 1000.0,
        "delta_Ef_Eh": delta_ef_eh,
        "delta_Er_Eh": delta_er_eh,
    }


def main() -> None:
    ch3cho = Species(
        name="CH3CHO",
        energy_hartree=-153.429498589980540,
        mass_amu=44.026215,
        freqs_cm=(
            166.3427, 499.2581, 771.3471, 896.1367, 1120.6587,
            1125.5271, 1370.982, 1453.3390, 1465.6330, 1802.8522,
            2907.9137, 3045.5309, 3126.2628, 3172.4558,
        ),
        rot_constants_cm=(0.2983800487, 0.3328657081, 1.8612519936),
        symmetry_number=1,
        electronic_degeneracy=1,
    )

    h_atom = Species(
        name="H",
        energy_hartree=-0.49982,
        mass_amu=1.00782503223,
        freqs_cm=(),
        rot_constants_cm=(),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    ts = Species(
        name="TS",
        energy_hartree=-153.9000,
        mass_amu=ch3cho.mass_amu + h_atom.mass_amu,
        freqs_cm=(
            135.0482, 171.6758, 265.6897, 300.2467, 491.6734,
            878.3455, 892.2855, 1053.6440, 1165.0740, 1275.6397,
            1353.1128, 1394.2242, 1450.8119, 1458.4285, 1885.4714,
            3051.0869, 3141.5794,
        ),
        rot_constants_cm=(0.2605435343, 0.3283089104, 1.0177594421),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    h2 = Species(
        name="H2",
        energy_hartree=-1.128746115957980,
        mass_amu=2.01565007,
        freqs_cm=(4583.9090,),
        rot_constants_cm=(59.7986200445,),
        symmetry_number=2,
        electronic_degeneracy=1,
    )

    ch3co = Species(
        name="CH3CO",
        energy_hartree=-152.784900719202085,
        mass_amu=43.0183897,
        freqs_cm=(
            457.0776, 847.6872, 879.7312, 1039.0968, 1336.8913,
            1445.1669, 1469.0428, 1893.7103, 3051.3571, 3146.0186,
            3185.0905,
        ),
        rot_constants_cm=(0.3099207198, 0.3286687762, 2.6476526610),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    imag_freq_cm = 1487.0405

    # A gorbe tartomanya es pontszama.
    temperature_min_k = 15.0
    temperature_max_k = 150.0
    number_of_temperature_points = 30

    if number_of_temperature_points < 2:
        raise ValueError("ket pont nelkul nincs egyenes kutya.")

    temperature_step_k = (
        temperature_max_k - temperature_min_k
    ) / (number_of_temperature_points - 1)

    temperatures = [
        temperature_min_k + i * temperature_step_k
        for i in range(number_of_temperature_points)
    ]

    k_tst_list = []
    k_wigner_list = []
    k_eckart_list = []
    kappa_wigner_list = []
    kappa_eckart_list = []

    for temperature_k in temperatures:
        res = canonical_tst_bimolecular_rate(
            reactant_a=ch3cho,
            reactant_b=h_atom,
            transition_state=ts,
            product_a=ch3co,
            product_b=h2,
            temperature_k=temperature_k,
            imag_freq_cm=imag_freq_cm,
        )

        k_tst_list.append(res["k_tst_cm3"])
        k_wigner_list.append(res["k_wigner_cm3"])
        k_eckart_list.append(res["k_corrected_cm3"])
        kappa_wigner_list.append(res["kappa_wigner"])
        kappa_eckart_list.append(res["kappa_eckart"])

    res_298 = canonical_tst_bimolecular_rate(
        ch3cho, h_atom, ts, ch3co, h2, 298.15, imag_freq_cm
    )

    print("=== EREDMENYEK (T = 298.15 K) ===")
    print(f"Forward gat, Delta Ef: {res_298['delta_Ef_kJ_mol']:.2f} kJ/mol")
    print(f"Reverse gat, Delta Er: {res_298['delta_Er_kJ_mol']:.2f} kJ/mol")
    print(f"Delta Ef: {res_298['delta_Ef_Eh']:.8f} Eh")
    print(f"Delta Er: {res_298['delta_Er_Eh']:.8f} Eh")
    print(f"Wigner kappa(298.15 K): {res_298['kappa_wigner']:.6g}")
    print(f"Eckart kappa(298.15 K): {res_298['kappa_eckart']:.6g}")
    print(f"k, klasszikus TST: {res_298['k_tst_cm3']:.6e} cm^3 molecule^-1 s^-1")
    print(
        f"k, Wigner-korrigalt: {res_298['k_wigner_cm3']:.6e} "
        "cm^3 molecule^-1 s^-1"
    )
    print(
        f"k, Eckart-korrigalt: {res_298['k_corrected_cm3']:.6e} "
        "cm^3 molecule^-1 s^-1"
    )

    show_comparison_plot(
        temperatures,
        k_tst_list,
        k_wigner_list,
        k_eckart_list,
        kappa_wigner_list,
        kappa_eckart_list,
    )


if __name__ == "__main__":
    main()