from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FloresExtractor"]
_LOG = setup_logger(__name__)

task_names = (
        "african_flores",
        "african_flores_tasks",
        "darija_translation_flores",
        "darija_translation_tasks_flores",
        "flores",
        "flores_ace_Arab-eng_Latn_prompt_1",
        "flores_ace_Arab-eng_Latn_prompt_2",
        "flores_ace_Arab-eng_Latn_prompt_3",
        "flores_ace_Latn-eng_Latn_prompt_1",
        "flores_ace_Latn-eng_Latn_prompt_2",
        "flores_ace_Latn-eng_Latn_prompt_3",
        "flores_acq_Arab-eng_Latn_prompt_1",
        "flores_acq_Arab-eng_Latn_prompt_2",
        "flores_acq_Arab-eng_Latn_prompt_3",
        "flores_aeb_Arab-eng_Latn_prompt_1",
        "flores_aeb_Arab-eng_Latn_prompt_2",
        "flores_aeb_Arab-eng_Latn_prompt_3",
        "flores_afr-eng",
        "flores_afr-eng_prompt_1",
        "flores_afr-eng_prompt_2",
        "flores_afr-eng_prompt_3",
        "flores_afr_Latn-eng_Latn_prompt_1",
        "flores_afr_Latn-eng_Latn_prompt_2",
        "flores_afr_Latn-eng_Latn_prompt_3",
        "flores_aka_Latn-eng_Latn_prompt_1",
        "flores_aka_Latn-eng_Latn_prompt_2",
        "flores_aka_Latn-eng_Latn_prompt_3",
        "flores_amh_Ethi-eng_Latn_prompt_1",
        "flores_amh_Ethi-eng_Latn_prompt_2",
        "flores_amh_Ethi-eng_Latn_prompt_3",
        "flores_ary_Arab-eng_Latn_prompt_1",
        "flores_ary_Arab-eng_Latn_prompt_2",
        "flores_ary_Arab-eng_Latn_prompt_3",
        "flores_arz_Arab-eng_Latn_prompt_1",
        "flores_arz_Arab-eng_Latn_prompt_2",
        "flores_arz_Arab-eng_Latn_prompt_3",
        "flores_bam_Latn-eng_Latn_prompt_1",
        "flores_bam_Latn-eng_Latn_prompt_2",
        "flores_bam_Latn-eng_Latn_prompt_3",
        "flores_ban_Latn-eng_Latn_prompt_1",
        "flores_ban_Latn-eng_Latn_prompt_2",
        "flores_ban_Latn-eng_Latn_prompt_3",
        "flores_bem_Latn-eng_Latn_prompt_1",
        "flores_bem_Latn-eng_Latn_prompt_2",
        "flores_bem_Latn-eng_Latn_prompt_3",
        "flores_ca",
        "flores_ca-de",
        "flores_ca-en",
        "flores_ca-es",
        "flores_ca-eu",
        "flores_ca-fr",
        "flores_ca-gl",
        "flores_ca-it",
        "flores_ca-pt",
        "flores_cjk_Latn-eng_Latn_prompt_1",
        "flores_cjk_Latn-eng_Latn_prompt_2",
        "flores_cjk_Latn-eng_Latn_prompt_3",
        "flores_de-ca",
        "flores_de-es",
        "flores_de-eu",
        "flores_de-gl",
        "flores_de-pt",
        "flores_dik_Latn-eng_Latn_prompt_1",
        "flores_dik_Latn-eng_Latn_prompt_2",
        "flores_dik_Latn-eng_Latn_prompt_3",
        "flores_dyu_Latn-eng_Latn_prompt_1",
        "flores_dyu_Latn-eng_Latn_prompt_2",
        "flores_dyu_Latn-eng_Latn_prompt_3",
        "flores_en-ca",
        "flores_en-es",
        "flores_en-eu",
        "flores_en-gl",
        "flores_en-pt",
        "flores_eng-afr",
        "flores_eng-afr_prompt_1",
        "flores_eng-afr_prompt_2",
        "flores_eng-afr_prompt_3",
        "flores_eng_Latn-ace_Arab_prompt_1",
        "flores_eng_Latn-ace_Arab_prompt_2",
        "flores_eng_Latn-ace_Arab_prompt_3",
        "flores_eng_Latn-ace_Latn_prompt_1",
        "flores_eng_Latn-ace_Latn_prompt_2",
        "flores_eng_Latn-ace_Latn_prompt_3",
        "flores_eng_Latn-acq_Arab_prompt_1",
        "flores_eng_Latn-acq_Arab_prompt_2",
        "flores_eng_Latn-acq_Arab_prompt_3",
        "flores_eng_Latn-aeb_Arab_prompt_1",
        "flores_eng_Latn-aeb_Arab_prompt_2",
        "flores_eng_Latn-aeb_Arab_prompt_3",
        "flores_eng_Latn-afr_Latn_prompt_1",
        "flores_eng_Latn-afr_Latn_prompt_2",
        "flores_eng_Latn-afr_Latn_prompt_3",
        "flores_eng_Latn-aka_Latn_prompt_1",
        "flores_eng_Latn-aka_Latn_prompt_2",
        "flores_eng_Latn-aka_Latn_prompt_3",
        "flores_eng_Latn-amh_Ethi_prompt_1",
        "flores_eng_Latn-amh_Ethi_prompt_2",
        "flores_eng_Latn-amh_Ethi_prompt_3",
        "flores_eng_Latn-ary_Arab_prompt_1",
        "flores_eng_Latn-ary_Arab_prompt_2",
        "flores_eng_Latn-ary_Arab_prompt_3",
        "flores_eng_Latn-arz_Arab_prompt_1",
        "flores_eng_Latn-arz_Arab_prompt_2",
        "flores_eng_Latn-arz_Arab_prompt_3",
        "flores_eng_Latn-bam_Latn_prompt_1",
        "flores_eng_Latn-bam_Latn_prompt_2",
        "flores_eng_Latn-bam_Latn_prompt_3",
        "flores_eng_Latn-ban_Latn_prompt_1",
        "flores_eng_Latn-ban_Latn_prompt_2",
        "flores_eng_Latn-ban_Latn_prompt_3",
        "flores_eng_Latn-bem_Latn_prompt_1",
        "flores_eng_Latn-bem_Latn_prompt_2",
        "flores_eng_Latn-bem_Latn_prompt_3",
        "flores_eng_Latn-cjk_Latn_prompt_1",
        "flores_eng_Latn-cjk_Latn_prompt_2",
        "flores_eng_Latn-cjk_Latn_prompt_3",
        "flores_eng_Latn-dik_Latn_prompt_1",
        "flores_eng_Latn-dik_Latn_prompt_2",
        "flores_eng_Latn-dik_Latn_prompt_3",
        "flores_eng_Latn-dyu_Latn_prompt_1",
        "flores_eng_Latn-dyu_Latn_prompt_2",
        "flores_eng_Latn-dyu_Latn_prompt_3",
        "flores_eng_Latn-ewe_Latn_prompt_1",
        "flores_eng_Latn-ewe_Latn_prompt_2",
        "flores_eng_Latn-ewe_Latn_prompt_3",
        "flores_eng_Latn-fon_Latn_prompt_1",
        "flores_eng_Latn-fon_Latn_prompt_2",
        "flores_eng_Latn-fon_Latn_prompt_3",
        "flores_eng_Latn-fra_Latn_prompt_1",
        "flores_eng_Latn-fra_Latn_prompt_2",
        "flores_eng_Latn-fra_Latn_prompt_3",
        "flores_eng_Latn-fuv_Latn_prompt_1",
        "flores_eng_Latn-fuv_Latn_prompt_2",
        "flores_eng_Latn-fuv_Latn_prompt_3",
        "flores_eng_Latn-gaz_Latn_prompt_1",
        "flores_eng_Latn-gaz_Latn_prompt_2",
        "flores_eng_Latn-gaz_Latn_prompt_3",
        "flores_eng_Latn-hau_Latn_prompt_1",
        "flores_eng_Latn-hau_Latn_prompt_2",
        "flores_eng_Latn-hau_Latn_prompt_3",
        "flores_eng_Latn-ibo_Latn_prompt_1",
        "flores_eng_Latn-ibo_Latn_prompt_2",
        "flores_eng_Latn-ibo_Latn_prompt_3",
        "flores_eng_Latn-kab_Latn_prompt_1",
        "flores_eng_Latn-kab_Latn_prompt_2",
        "flores_eng_Latn-kab_Latn_prompt_3",
        "flores_eng_Latn-kam_Latn_prompt_1",
        "flores_eng_Latn-kam_Latn_prompt_2",
        "flores_eng_Latn-kam_Latn_prompt_3",
        "flores_eng_Latn-kbp_Latn_prompt_1",
        "flores_eng_Latn-kbp_Latn_prompt_2",
        "flores_eng_Latn-kbp_Latn_prompt_3",
        "flores_eng_Latn-kea_Latn_prompt_1",
        "flores_eng_Latn-kea_Latn_prompt_2",
        "flores_eng_Latn-kea_Latn_prompt_3",
        "flores_eng_Latn-kik_Latn_prompt_1",
        "flores_eng_Latn-kik_Latn_prompt_2",
        "flores_eng_Latn-kik_Latn_prompt_3",
        "flores_eng_Latn-kin_Latn_prompt_1",
        "flores_eng_Latn-kin_Latn_prompt_2",
        "flores_eng_Latn-kin_Latn_prompt_3",
        "flores_eng_Latn-kmb_Latn_prompt_1",
        "flores_eng_Latn-kmb_Latn_prompt_2",
        "flores_eng_Latn-kmb_Latn_prompt_3",
        "flores_eng_Latn-knc_Arab_prompt_1",
        "flores_eng_Latn-knc_Arab_prompt_2",
        "flores_eng_Latn-knc_Arab_prompt_3",
        "flores_eng_Latn-knc_Latn_prompt_1",
        "flores_eng_Latn-knc_Latn_prompt_2",
        "flores_eng_Latn-knc_Latn_prompt_3",
        "flores_eng_Latn-kon_Latn_prompt_1",
        "flores_eng_Latn-kon_Latn_prompt_2",
        "flores_eng_Latn-kon_Latn_prompt_3",
        "flores_eng_Latn-lin_Latn_prompt_1",
        "flores_eng_Latn-lin_Latn_prompt_2",
        "flores_eng_Latn-lin_Latn_prompt_3",
        "flores_eng_Latn-lua_Latn_prompt_1",
        "flores_eng_Latn-lua_Latn_prompt_2",
        "flores_eng_Latn-lua_Latn_prompt_3",
        "flores_eng_Latn-lug_Latn_prompt_1",
        "flores_eng_Latn-lug_Latn_prompt_2",
        "flores_eng_Latn-lug_Latn_prompt_3",
        "flores_eng_Latn-luo_Latn_prompt_1",
        "flores_eng_Latn-luo_Latn_prompt_2",
        "flores_eng_Latn-luo_Latn_prompt_3",
        "flores_eng_Latn-mos_Latn_prompt_1",
        "flores_eng_Latn-mos_Latn_prompt_2",
        "flores_eng_Latn-mos_Latn_prompt_3",
        "flores_eng_Latn-nso_Latn_prompt_1",
        "flores_eng_Latn-nso_Latn_prompt_2",
        "flores_eng_Latn-nso_Latn_prompt_3",
        "flores_eng_Latn-nus_Latn_prompt_1",
        "flores_eng_Latn-nus_Latn_prompt_2",
        "flores_eng_Latn-nus_Latn_prompt_3",
        "flores_eng_Latn-nya_Latn_prompt_1",
        "flores_eng_Latn-nya_Latn_prompt_2",
        "flores_eng_Latn-nya_Latn_prompt_3",
        "flores_eng_Latn-plt_Latn_prompt_1",
        "flores_eng_Latn-plt_Latn_prompt_2",
        "flores_eng_Latn-plt_Latn_prompt_3",
        "flores_eng_Latn-run_Latn_prompt_1",
        "flores_eng_Latn-run_Latn_prompt_2",
        "flores_eng_Latn-run_Latn_prompt_3",
        "flores_eng_Latn-sag_Latn_prompt_1",
        "flores_eng_Latn-sag_Latn_prompt_2",
        "flores_eng_Latn-sag_Latn_prompt_3",
        "flores_eng_Latn-sna_Latn_prompt_1",
        "flores_eng_Latn-sna_Latn_prompt_2",
        "flores_eng_Latn-sna_Latn_prompt_3",
        "flores_eng_Latn-som_Latn_prompt_1",
        "flores_eng_Latn-som_Latn_prompt_2",
        "flores_eng_Latn-som_Latn_prompt_3",
        "flores_eng_Latn-sot_Latn_prompt_1",
        "flores_eng_Latn-sot_Latn_prompt_2",
        "flores_eng_Latn-sot_Latn_prompt_3",
        "flores_eng_Latn-ssw_Latn_prompt_1",
        "flores_eng_Latn-ssw_Latn_prompt_2",
        "flores_eng_Latn-ssw_Latn_prompt_3",
        "flores_eng_Latn-sun_Latn_prompt_1",
        "flores_eng_Latn-sun_Latn_prompt_2",
        "flores_eng_Latn-sun_Latn_prompt_3",
        "flores_eng_Latn-swh_Latn_prompt_1",
        "flores_eng_Latn-swh_Latn_prompt_2",
        "flores_eng_Latn-swh_Latn_prompt_3",
        "flores_eng_Latn-taq_Latn_prompt_1",
        "flores_eng_Latn-taq_Latn_prompt_2",
        "flores_eng_Latn-taq_Latn_prompt_3",
        "flores_eng_Latn-taq_Tfng_prompt_1",
        "flores_eng_Latn-taq_Tfng_prompt_2",
        "flores_eng_Latn-taq_Tfng_prompt_3",
        "flores_eng_Latn-tir_Ethi_prompt_1",
        "flores_eng_Latn-tir_Ethi_prompt_2",
        "flores_eng_Latn-tir_Ethi_prompt_3",
        "flores_eng_Latn-tsn_Latn_prompt_1",
        "flores_eng_Latn-tsn_Latn_prompt_2",
        "flores_eng_Latn-tsn_Latn_prompt_3",
        "flores_eng_Latn-tso_Latn_prompt_1",
        "flores_eng_Latn-tso_Latn_prompt_2",
        "flores_eng_Latn-tso_Latn_prompt_3",
        "flores_eng_Latn-tum_Latn_prompt_1",
        "flores_eng_Latn-tum_Latn_prompt_2",
        "flores_eng_Latn-tum_Latn_prompt_3",
        "flores_eng_Latn-twi_Latn_prompt_1",
        "flores_eng_Latn-twi_Latn_prompt_2",
        "flores_eng_Latn-twi_Latn_prompt_3",
        "flores_eng_Latn-tzm_Tfng_prompt_1",
        "flores_eng_Latn-tzm_Tfng_prompt_2",
        "flores_eng_Latn-tzm_Tfng_prompt_3",
        "flores_eng_Latn-umb_Latn_prompt_1",
        "flores_eng_Latn-umb_Latn_prompt_2",
        "flores_eng_Latn-umb_Latn_prompt_3",
        "flores_eng_Latn-wol_Latn_prompt_1",
        "flores_eng_Latn-wol_Latn_prompt_2",
        "flores_eng_Latn-wol_Latn_prompt_3",
        "flores_eng_Latn-xho_Latn_prompt_1",
        "flores_eng_Latn-xho_Latn_prompt_2",
        "flores_eng_Latn-xho_Latn_prompt_3",
        "flores_eng_Latn-yor_Latn_prompt_1",
        "flores_eng_Latn-yor_Latn_prompt_2",
        "flores_eng_Latn-yor_Latn_prompt_3",
        "flores_eng_Latn-zul_Latn_prompt_1",
        "flores_eng_Latn-zul_Latn_prompt_2",
        "flores_eng_Latn-zul_Latn_prompt_3",
        "flores_es",
        "flores_es-ca",
        "flores_es-de",
        "flores_es-en",
        "flores_es-eu",
        "flores_es-fr",
        "flores_es-gl",
        "flores_es-it",
        "flores_es-pt",
        "flores_eu",
        "flores_eu-ca",
        "flores_eu-de",
        "flores_eu-en",
        "flores_eu-es",
        "flores_eu-fr",
        "flores_eu-gl",
        "flores_eu-it",
        "flores_eu-pt",
        "flores_ewe_Latn-eng_Latn_prompt_1",
        "flores_ewe_Latn-eng_Latn_prompt_2",
        "flores_ewe_Latn-eng_Latn_prompt_3",
        "flores_fon_Latn-eng_Latn_prompt_1",
        "flores_fon_Latn-eng_Latn_prompt_2",
        "flores_fon_Latn-eng_Latn_prompt_3",
        "flores_fr-ca",
        "flores_fr-es",
        "flores_fr-eu",
        "flores_fr-gl",
        "flores_fr-pt",
        "flores_fra_Latn-eng_Latn_prompt_1",
        "flores_fra_Latn-eng_Latn_prompt_2",
        "flores_fra_Latn-eng_Latn_prompt_3",
        "flores_fuv_Latn-eng_Latn_prompt_1",
        "flores_fuv_Latn-eng_Latn_prompt_2",
        "flores_fuv_Latn-eng_Latn_prompt_3",
        "flores_gaz_Latn-eng_Latn_prompt_1",
        "flores_gaz_Latn-eng_Latn_prompt_2",
        "flores_gaz_Latn-eng_Latn_prompt_3",
        "flores_gl",
        "flores_gl-ca",
        "flores_gl-de",
        "flores_gl-en",
        "flores_gl-es",
        "flores_gl-eu",
        "flores_gl-fr",
        "flores_gl-it",
        "flores_gl-pt",
        "flores_hau_Latn-eng_Latn_prompt_1",
        "flores_hau_Latn-eng_Latn_prompt_2",
        "flores_hau_Latn-eng_Latn_prompt_3",
        "flores_ibo_Latn-eng_Latn_prompt_1",
        "flores_ibo_Latn-eng_Latn_prompt_2",
        "flores_ibo_Latn-eng_Latn_prompt_3",
        "flores_it-ca",
        "flores_it-es",
        "flores_it-eu",
        "flores_it-gl",
        "flores_it-pt",
        "flores_kab_Latn-eng_Latn_prompt_1",
        "flores_kab_Latn-eng_Latn_prompt_2",
        "flores_kab_Latn-eng_Latn_prompt_3",
        "flores_kam_Latn-eng_Latn_prompt_1",
        "flores_kam_Latn-eng_Latn_prompt_2",
        "flores_kam_Latn-eng_Latn_prompt_3",
        "flores_kbp_Latn-eng_Latn_prompt_1",
        "flores_kbp_Latn-eng_Latn_prompt_2",
        "flores_kbp_Latn-eng_Latn_prompt_3",
        "flores_kea_Latn-eng_Latn_prompt_1",
        "flores_kea_Latn-eng_Latn_prompt_2",
        "flores_kea_Latn-eng_Latn_prompt_3",
        "flores_kik_Latn-eng_Latn_prompt_1",
        "flores_kik_Latn-eng_Latn_prompt_2",
        "flores_kik_Latn-eng_Latn_prompt_3",
        "flores_kin_Latn-eng_Latn_prompt_1",
        "flores_kin_Latn-eng_Latn_prompt_2",
        "flores_kin_Latn-eng_Latn_prompt_3",
        "flores_kmb_Latn-eng_Latn_prompt_1",
        "flores_kmb_Latn-eng_Latn_prompt_2",
        "flores_kmb_Latn-eng_Latn_prompt_3",
        "flores_knc_Arab-eng_Latn_prompt_1",
        "flores_knc_Arab-eng_Latn_prompt_2",
        "flores_knc_Arab-eng_Latn_prompt_3",
        "flores_knc_Latn-eng_Latn_prompt_1",
        "flores_knc_Latn-eng_Latn_prompt_2",
        "flores_knc_Latn-eng_Latn_prompt_3",
        "flores_kon_Latn-eng_Latn_prompt_1",
        "flores_kon_Latn-eng_Latn_prompt_2",
        "flores_kon_Latn-eng_Latn_prompt_3",
        "flores_lin_Latn-eng_Latn_prompt_1",
        "flores_lin_Latn-eng_Latn_prompt_2",
        "flores_lin_Latn-eng_Latn_prompt_3",
        "flores_lua_Latn-eng_Latn_prompt_1",
        "flores_lua_Latn-eng_Latn_prompt_2",
        "flores_lua_Latn-eng_Latn_prompt_3",
        "flores_lug_Latn-eng_Latn_prompt_1",
        "flores_lug_Latn-eng_Latn_prompt_2",
        "flores_lug_Latn-eng_Latn_prompt_3",
        "flores_luo_Latn-eng_Latn_prompt_1",
        "flores_luo_Latn-eng_Latn_prompt_2",
        "flores_luo_Latn-eng_Latn_prompt_3",
        "flores_mos_Latn-eng_Latn_prompt_1",
        "flores_mos_Latn-eng_Latn_prompt_2",
        "flores_mos_Latn-eng_Latn_prompt_3",
        "flores_nso_Latn-eng_Latn_prompt_1",
        "flores_nso_Latn-eng_Latn_prompt_2",
        "flores_nso_Latn-eng_Latn_prompt_3",
        "flores_nus_Latn-eng_Latn_prompt_1",
        "flores_nus_Latn-eng_Latn_prompt_2",
        "flores_nus_Latn-eng_Latn_prompt_3",
        "flores_nya_Latn-eng_Latn_prompt_1",
        "flores_nya_Latn-eng_Latn_prompt_2",
        "flores_nya_Latn-eng_Latn_prompt_3",
        "flores_plt_Latn-eng_Latn_prompt_1",
        "flores_plt_Latn-eng_Latn_prompt_2",
        "flores_plt_Latn-eng_Latn_prompt_3",
        "flores_pt",
        "flores_pt-ca",
        "flores_pt-de",
        "flores_pt-en",
        "flores_pt-es",
        "flores_pt-eu",
        "flores_pt-fr",
        "flores_pt-gl",
        "flores_pt-it",
        "flores_run_Latn-eng_Latn_prompt_1",
        "flores_run_Latn-eng_Latn_prompt_2",
        "flores_run_Latn-eng_Latn_prompt_3",
        "flores_sag_Latn-eng_Latn_prompt_1",
        "flores_sag_Latn-eng_Latn_prompt_2",
        "flores_sag_Latn-eng_Latn_prompt_3",
        "flores_sna_Latn-eng_Latn_prompt_1",
        "flores_sna_Latn-eng_Latn_prompt_2",
        "flores_sna_Latn-eng_Latn_prompt_3",
        "flores_som_Latn-eng_Latn_prompt_1",
        "flores_som_Latn-eng_Latn_prompt_2",
        "flores_som_Latn-eng_Latn_prompt_3",
        "flores_sot_Latn-eng_Latn_prompt_1",
        "flores_sot_Latn-eng_Latn_prompt_2",
        "flores_sot_Latn-eng_Latn_prompt_3",
        "flores_ssw_Latn-eng_Latn_prompt_1",
        "flores_ssw_Latn-eng_Latn_prompt_2",
        "flores_ssw_Latn-eng_Latn_prompt_3",
        "flores_sun_Latn-eng_Latn_prompt_1",
        "flores_sun_Latn-eng_Latn_prompt_2",
        "flores_sun_Latn-eng_Latn_prompt_3",
        "flores_swh_Latn-eng_Latn_prompt_1",
        "flores_swh_Latn-eng_Latn_prompt_2",
        "flores_swh_Latn-eng_Latn_prompt_3",
        "flores_taq_Latn-eng_Latn_prompt_1",
        "flores_taq_Latn-eng_Latn_prompt_2",
        "flores_taq_Latn-eng_Latn_prompt_3",
        "flores_taq_Tfng-eng_Latn_prompt_1",
        "flores_taq_Tfng-eng_Latn_prompt_2",
        "flores_taq_Tfng-eng_Latn_prompt_3",
        "flores_tir_Ethi-eng_Latn_prompt_1",
        "flores_tir_Ethi-eng_Latn_prompt_2",
        "flores_tir_Ethi-eng_Latn_prompt_3",
        "flores_tsn_Latn-eng_Latn_prompt_1",
        "flores_tsn_Latn-eng_Latn_prompt_2",
        "flores_tsn_Latn-eng_Latn_prompt_3",
        "flores_tso_Latn-eng_Latn_prompt_1",
        "flores_tso_Latn-eng_Latn_prompt_2",
        "flores_tso_Latn-eng_Latn_prompt_3",
        "flores_tum_Latn-eng_Latn_prompt_1",
        "flores_tum_Latn-eng_Latn_prompt_2",
        "flores_tum_Latn-eng_Latn_prompt_3",
        "flores_twi_Latn-eng_Latn_prompt_1",
        "flores_twi_Latn-eng_Latn_prompt_2",
        "flores_twi_Latn-eng_Latn_prompt_3",
        "flores_tzm_Tfng-eng_Latn_prompt_1",
        "flores_tzm_Tfng-eng_Latn_prompt_2",
        "flores_tzm_Tfng-eng_Latn_prompt_3",
        "flores_umb_Latn-eng_Latn_prompt_1",
        "flores_umb_Latn-eng_Latn_prompt_2",
        "flores_umb_Latn-eng_Latn_prompt_3",
        "flores_wol_Latn-eng_Latn_prompt_1",
        "flores_wol_Latn-eng_Latn_prompt_2",
        "flores_wol_Latn-eng_Latn_prompt_3",
        "flores_xho_Latn-eng_Latn_prompt_1",
        "flores_xho_Latn-eng_Latn_prompt_2",
        "flores_xho_Latn-eng_Latn_prompt_3",
        "flores_yor_Latn-eng_Latn_prompt_1",
        "flores_yor_Latn-eng_Latn_prompt_2",
        "flores_yor_Latn-eng_Latn_prompt_3",
        "flores_zul_Latn-eng_Latn_prompt_1",
        "flores_zul_Latn-eng_Latn_prompt_2",
        "flores_zul_Latn-eng_Latn_prompt_3",
        "trasnlation_all_flores",
        "trasnlation_dr_en_flores",
        "trasnlation_dr_fr_flores",
        "trasnlation_dr_msa_flores",
        "trasnlation_en_dr_flores",
        "trasnlation_fr_dr_flores",
        "trasnlation_msa_dr_flores",
    )

class FloresExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Flores benchmark - multilingual machine translation tasks."""


    evaluator_name = "generation"
    
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="flores")
        max_items = self._normalize_limit(limit)
        
        # Load data directly from HuggingFace
        from datasets import load_dataset
        try:
            # Try to load from cache (trust_remote_code no longer supported)
            ds = load_dataset("facebook/flores", "all", split="devtest")
            docs = list(ds)
            if max_items:
                docs = docs[:max_items]
        except Exception as e:
            log.error(f"Failed to load flores dataset: {e}")
            return []
        
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Flores doc into a ContrastivePair.

        Flores format:
        - sentence_{source_lang}_{script}: source text
        - sentence_{target_lang}_{script}: target text

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Find sentence fields (format: sentence_{lang}_{script})
            sentence_fields = [k for k in doc.keys() if k.startswith("sentence_")]

            if len(sentence_fields) < 2:
                log.debug("Skipping doc due to missing sentence fields", extra={"doc": doc})
                return None

            # Get source and target sentences
            # Usually first is source, second is target
            source_field = sentence_fields[0]
            target_field = sentence_fields[1]

            source_text = doc.get(source_field, "").strip()
            target_text = doc.get(target_field, "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to empty text", extra={"doc": doc})
                return None

            # Extract language codes for prompt
            # Format: sentence_afr_Latn → afr_Latn
            source_lang = source_field.replace("sentence_", "")
            target_lang = target_field.replace("sentence_", "")

            # Create translation prompt
            prompt = f"Translate the following from {source_lang} to {target_lang}:\n{source_text}"

            # Positive: correct translation
            correct_translation = target_text

            # Negative: shuffled words for synthetic incorrect translation
            words = target_text.split()
            if len(words) < 2:
                # For single-word translations, use a placeholder
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = ' '.join(shuffled_words)

            metadata = {"label": "flores"}

            return self._build_pair(
                question=prompt,
                correct=correct_translation,
                incorrect=incorrect_translation,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

