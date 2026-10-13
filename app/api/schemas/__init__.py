from typing import Optional

from pydantic import BaseModel, Field


class ConstructionPredictionRequest(BaseModel):

    # ---------------------------------------------------------
    # VARIABLES CATEGÓRICAS
    # ---------------------------------------------------------

    codcantf: int = Field(
        ...,
        description="Código del cantón"
    )

    careaur: int = Field(
        ...,
        description="Área urbana o rural"
    )

    CTIPOBR: int = Field(
        ...,
        description="Tipo de obra"
    )

    cimi: int = Field(
        ...,
        description="Material predominante de cimientos"
    )

    piso: int = Field(
        ...,
        description="Material predominante de piso"
    )

    estru: int = Field(
        ...,
        description="Material predominante de estructura"
    )

    pared: int = Field(
        ...,
        description="Material predominante de paredes"
    )

    cubi: int = Field(
        ...,
        description="Material predominante de cubierta"
    )

    CASAEDIF: int = Field(
        ...,
        description="Tipo casa/edificio"
    )

    CDISPUSO: int = Field(
        ...,
        description="Uso de la edificación"
    )

    mes: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mes de referencia"
    )

    TOCUP: int = Field(
        ...,
        description="Tipo de ocupación"
    )

    VALTERR: int = Field(
        ...,
        description="Rango del valor del terreno"
    )

    # ---------------------------------------------------------
    # VARIABLES NUMÉRICAS
    # ---------------------------------------------------------

    NUPICAL: float = Field(
        ...,
        gt=0,
        description="Número de pisos"
    )

    CSUTE: float = Field(
        ...,
        ge=0,
        description="Superficie total del terreno"
    )

    CARCO: float = Field(
        ...,
        gt=0,
        description="Área total a construirse"
    )

    # Estas dos variables pueden no estar disponibles.
    # Si llegan como None, el Pipeline utilizará
    # el SimpleImputer entrenado.
    NUCUCAL: Optional[float] = Field(
        default=None,
        description="Número de cuartos"
    )

    NUDOCAL: Optional[float] = Field(
        default=None,
        description="Número de dormitorios"
    )