import logging
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.models import ProphetComponent


logger = logging.getLogger(__name__)


async def get_prophet_components_from_db(
    db: AsyncSession,
    start_ts: datetime,
    end_ts: datetime,
) -> pd.DataFrame | None:
    """
    Retrieve cached prophet components from database for a timestamp range.
    
    Returns None if no cached components exist for the range.
    """
    query = select(ProphetComponent).where(
        ProphetComponent.timestamp >= start_ts,
        ProphetComponent.timestamp <= end_ts,
    )
    result = await db.execute(query)
    rows = result.scalars().all()
    
    if not rows:
        return None
    
    # Convert to DataFrame for merging
    data = [
        {
            "timestamp": row.timestamp,
            "component_type": row.component_type,
            "value": row.value,
        }
        for row in rows
    ]
    
    component_df = pd.DataFrame(data)
    logger.info(
        "Retrieved %d cached prophet components from database (%s to %s)",
        len(data),
        start_ts,
        end_ts,
    )
    return component_df


async def fit_and_cache_prophet_components(
    db: AsyncSession,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit Prophet model to data and cache components in database.
    
    Returns DataFrame with added prophet component columns.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not available, skipping component fitting")
        result = df.copy()
        for col in ["prophet_trend", "prophet_yearly", "prophet_weekly", "prophet_daily"]:
            result[col] = 0.0
        return result
    
    component_cols = ["prophet_trend", "prophet_yearly", "prophet_weekly", "prophet_daily"]
    
    if df.empty or len(df) < 24:
        logger.warning("Not enough data for prophet fitting (need >= 24 rows, got %d)", len(df))
        result = df.copy()
        for col in component_cols:
            result[col] = 0.0
        return result
    
    # Prepare data for Prophet
    prophet_df = df[["Timestamp", "Ontario Demand"]].copy()
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df.rename(columns={"Timestamp": "ds", "Ontario Demand": "y"})
    
    if len(prophet_df) < 24:
        logger.warning("Not enough valid data for prophet fitting")
        result = df.copy()
        for col in component_cols:
            result[col] = 0.0
        return result
    
    # Fit Prophet model
    logger.info("Fitting Prophet model with %d rows", len(prophet_df))
    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)
    
    # Generate predictions
    components = model.predict(prophet_df[["ds"]])
    
    # Get date range for caching metadata
    train_start_date = pd.to_datetime(prophet_df["ds"].min()).date()
    train_end_date = pd.to_datetime(prophet_df["ds"].max()).date()
    train_row_count = len(prophet_df)
    
    # Cache components in database
    fitted_at = datetime.now(timezone.utc)
    components_to_cache = []
    
    for component_type in component_cols:
        for idx, row in components.iterrows():
            ts = pd.to_datetime(row["ds"])
            value = float(row.get(component_type.replace("prophet_", ""), 0.0))
            
            component = ProphetComponent(
                fitted_at=fitted_at,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                train_row_count=train_row_count,
                component_type=component_type,
                timestamp=ts,
                value=value,
            )
            components_to_cache.append(component)
    
    # Bulk insert
    db.add_all(components_to_cache)
    await db.commit()
    logger.info("Cached %d prophet components to database", len(components_to_cache))
    
    # Create result DataFrame
    component_df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(prophet_df["ds"]).values,
            "prophet_trend": components.get("trend", 0.0).values,
            "prophet_yearly": components.get("yearly", 0.0).values,
            "prophet_weekly": components.get("weekly", 0.0).values,
            "prophet_daily": components.get("daily", 0.0).values,
        }
    )
    
    result = pd.merge(df.copy(), component_df, on="Timestamp", how="left")
    for col in component_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    
    return result


async def are_prophet_components_stale(
    db: AsyncSession,
    df: pd.DataFrame,
) -> bool:
    """
    Check if cached prophet components are stale compared to current data.
    
    Considers components stale if:
    - No cached components exist
    - Data extends beyond cached range
    - Row count has increased by PROPHET_CACHE_STALE_ROWS or more
    """
    import os
    PROPHET_CACHE_STALE_ROWS = int(os.getenv("PROPHET_CACHE_STALE_ROWS", "168"))
    
    if df.empty:
        return True
    
    # Get timestamp range from data
    ts_min = pd.to_datetime(df["Timestamp"].min())
    ts_max = pd.to_datetime(df["Timestamp"].max())
    current_row_count = len(df)
    
    # Query for any cached components
    query = select(ProphetComponent).limit(1)
    result = await db.execute(query)
    cached = result.scalar_one_or_none()
    
    if cached is None:
        logger.info("No cached prophet components found")
        return True
    
    # Check if data extends beyond cache
    if ts_min < pd.Timestamp(cached.train_start_date, tz=timezone.utc):
        logger.info("Data extends before cached range, components are stale")
        return True
    
    if ts_max > pd.Timestamp(cached.train_end_date, tz=timezone.utc):
        logger.info("Data extends after cached range, components are stale")
        return True
    
    # Check if row count increased by threshold or more
    stale_rows = current_row_count - cached.train_row_count
    if stale_rows >= PROPHET_CACHE_STALE_ROWS:
        logger.info(
            "Row count increased by %d rows (threshold: %d), components are stale",
            stale_rows,
            PROPHET_CACHE_STALE_ROWS,
        )
        return True
    
    logger.info("Cached prophet components are still fresh")
    return False


async def add_prophet_components_cached(
    db: AsyncSession,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add prophet components to DataFrame, using cache when possible.
    
    First checks if cached components are valid. If so, retrieves them from DB.
    Otherwise, fits new model and caches components.
    """
    component_cols = ["prophet_trend", "prophet_yearly", "prophet_weekly", "prophet_daily"]
    
    # Check if data is too small
    if df.empty or len(df) < 24:
        logger.warning("Insufficient data for prophet components")
        result = df.copy()
        for col in component_cols:
            result[col] = 0.0
        return result
    
    # Check if cached components are stale
    stale = await are_prophet_components_stale(db, df)
    
    if not stale:
        # Try to retrieve from cache
        ts_min = pd.to_datetime(df["Timestamp"].min())
        ts_max = pd.to_datetime(df["Timestamp"].max())
        
        component_df = await get_prophet_components_from_db(db, ts_min, ts_max)
        
        if component_df is not None:
            # Pivot to wide format for merging
            pivot_df = component_df.pivot_table(
                index="timestamp",
                columns="component_type",
                values="value",
                aggfunc="first",
            )
            pivot_df = pivot_df.reset_index()
            pivot_df = pivot_df.rename(columns={"timestamp": "Timestamp"})
            
            # Merge with original data
            result = pd.merge(df.copy(), pivot_df, on="Timestamp", how="left")
            
            # Fill any missing with zeros
            for col in component_cols:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
                else:
                    result[col] = 0.0
            
            return result
    
    # Fit new model and cache
    logger.info("Fitting and caching new prophet components")
    return await fit_and_cache_prophet_components(db, df)
