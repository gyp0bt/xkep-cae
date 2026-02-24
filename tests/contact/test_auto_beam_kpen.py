"""梁要素用 k_pen 自動推定テスト.

auto_beam_penalty_stiffness() の単体テスト。
EI/L³ ベースのペナルティ剛性推定が妥当であることを検証する。
"""

import math

import pytest

from xkep_cae.contact.law_normal import auto_beam_penalty_stiffness


class TestAutoBeamPenaltyStiffness:
    """auto_beam_penalty_stiffness() の単体テスト."""

    _E = 200e9  # Pa
    _I = 1e-12  # m⁴（直径2mmの円形断面の目安）
    _L = 0.01  # m（10mm要素長）

    def test_basic_value(self):
        """基本的な推定値が EI/L³ の妥当な倍数."""
        k = auto_beam_penalty_stiffness(self._E, self._I, self._L)
        k_bend = 12.0 * self._E * self._I / self._L**3
        # デフォルト scale=0.1, n_pairs=1 → k = 0.1 * k_bend
        expected = 0.1 * k_bend
        assert abs(k - expected) / expected < 1e-10

    def test_positive_value(self):
        """推定値は常に正."""
        k = auto_beam_penalty_stiffness(self._E, self._I, self._L)
        assert k > 0.0

    def test_scale_factor(self):
        """scale パラメータが線形に効く."""
        k1 = auto_beam_penalty_stiffness(self._E, self._I, self._L, scale=5.0)
        k2 = auto_beam_penalty_stiffness(self._E, self._I, self._L, scale=20.0)
        assert abs(k2 / k1 - 4.0) < 1e-10

    def test_n_pairs_scaling(self):
        """接触ペア数が増えると k_pen が低下する（線形スケーリング）."""
        k1 = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=1)
        k4 = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=4)
        # k4 = k1 / 4
        assert abs(k4 - k1 / 4.0) < 1e-6

    def test_n_pairs_monotone_decreasing(self):
        """ペア数が増えると k_pen は単調減少."""
        values = []
        for n in [1, 2, 4, 8, 16, 32]:
            k = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=n)
            values.append(k)
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], (
                f"n={2**i}: {values[i]} >= n={2 ** (i + 1)}: {values[i + 1]}"
            )

    def test_n_pairs_zero_treated_as_one(self):
        """n_contact_pairs=0 は 1 として扱われる."""
        k0 = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=0)
        k1 = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=1)
        assert abs(k0 - k1) < 1e-10

    def test_n_pairs_negative_treated_as_one(self):
        """n_contact_pairs < 0 は 1 として扱われる."""
        k_neg = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=-5)
        k1 = auto_beam_penalty_stiffness(self._E, self._I, self._L, n_contact_pairs=1)
        assert abs(k_neg - k1) < 1e-10

    def test_longer_element_reduces_kpen(self):
        """要素長が長いと k_pen が低下（L³ に反比例）."""
        k_short = auto_beam_penalty_stiffness(self._E, self._I, 0.005)
        k_long = auto_beam_penalty_stiffness(self._E, self._I, 0.010)
        # L が2倍 → k_pen は 1/8
        ratio = k_long / k_short
        assert abs(ratio - 1.0 / 8.0) < 1e-10

    def test_invalid_L_raises(self):
        """L_elem <= 0 で ValueError."""
        with pytest.raises(ValueError):
            auto_beam_penalty_stiffness(self._E, self._I, 0.0)
        with pytest.raises(ValueError):
            auto_beam_penalty_stiffness(self._E, self._I, -0.01)

    def test_typical_steel_wire(self):
        """鋼線（d=2mm, 16要素/40mm pitch）の推定値が妥当な範囲."""
        d = 0.002
        I_circ = math.pi * d**4 / 64
        L_elem = 0.040 / 16  # 2.5mm 要素長
        E = 200e9

        k = auto_beam_penalty_stiffness(E, I_circ, L_elem)
        k_bend = 12.0 * E * I_circ / L_elem**3
        # k = 0.1 * k_bend （n_pairs=1 デフォルト）
        assert k > 0
        ratio = k / k_bend
        assert 0.05 < ratio < 0.5, f"k/k_bend = {ratio:.4f}"

    def test_with_many_pairs_steel_wire(self):
        """多ペア（24ペア、7本撚り想定）の k_pen が過大でない."""
        d = 0.002
        I_circ = math.pi * d**4 / 64
        L_elem = 0.040 / 16
        E = 200e9

        k_single = auto_beam_penalty_stiffness(E, I_circ, L_elem, n_contact_pairs=1)
        k_multi = auto_beam_penalty_stiffness(E, I_circ, L_elem, n_contact_pairs=24)
        # 24ペア → k_multi = k_single / 24
        ratio = k_single / k_multi
        assert abs(ratio - 24.0) < 1e-6

    def test_formula_exact(self):
        """推定式の厳密な値を検証."""
        E, I_val, L = 100e9, 2e-12, 0.005
        n = 6
        s = 8.0
        k = auto_beam_penalty_stiffness(E, I_val, L, n_contact_pairs=n, scale=s)
        expected = s * 12.0 * E * I_val / L**3 / n
        assert abs(k - expected) < 1e-6

    def test_multi_contact_reasonable_magnitude(self):
        """120ペア（3本撚り16要素）で k_pen ≈ 1e5 オーダー."""
        d = 0.002
        I_circ = math.pi * d**4 / 64
        L_elem = 0.040 / 16  # 2.5mm
        E = 200e9

        k = auto_beam_penalty_stiffness(E, I_circ, L_elem, n_contact_pairs=120)
        # 過大（>1e8）でも過小（<1e3）でもないこと
        assert 1e3 < k < 1e8, f"k_pen = {k:.2e} は 1e3〜1e8 の範囲外"
