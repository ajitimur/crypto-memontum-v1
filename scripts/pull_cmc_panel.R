#!/usr/bin/env Rscript
#
# The one-time CoinMarketCap pull (ADR-0008).
#
# Emits a survivorship-free market-capitalisation panel as CSV. `crypto2` reads
# CoinMarketCap's `listings/historical` endpoint, which returns every asset
# listed as of each snapshot including the ones that later died; the documented
# `cryptocurrency/historical` endpoint returns zero rows for delisted coins and
# would bias every Universe built on it.
#
# Run this once. `crypto_momentum.data.cmc_panel.pull_panel` will not call it
# again once the CSV is in `data/raw/`, and it should not be called by hand
# either: the endpoints are undocumented, the route breaches CoinMarketCap's
# terms, and the practical exposure is an IP ban.
#
#   Rscript scripts/pull_cmc_panel.R --start 2013-04-28 --end 2026-08-31 \
#       --interval 7d --out data/raw/coinmarketcap/cmc-listings-historical.csv
#
# --end is required and has no default. Reading the clock here would make the
# window unpinnable and the pull unrepeatable; the caller passes the date it
# already stamped the manifest with.
#
# Needs: install.packages("crypto2")

suppressPackageStartupMessages({
  library(crypto2)
})

# The columns the Python side parses. Emitting anything else is a parse failure
# there rather than a silent column shift, which is the point of pinning them.
PANEL_COLUMNS <- c(
  "ts_utc", "cmc_id", "symbol", "name", "cmc_rank",
  "price_usd", "market_cap_usd", "volume_24h_usd", "circulating_supply"
)

parse_args <- function(argv) {
  defaults <- list(
    start = "2013-04-28",
    end = NA_character_,
    interval = "7d",
    out = "data/raw/coinmarketcap/cmc-listings-historical.csv"
  )
  index <- 1
  while (index < length(argv)) {
    key <- sub("^--", "", argv[[index]])
    if (!key %in% names(defaults)) {
      stop("unknown argument --", key, call. = FALSE)
    }
    defaults[[key]] <- argv[[index + 1]]
    index <- index + 2
  }
  if (is.na(defaults$end)) {
    stop(
      "--end is required: the window has to be pinned for the pull to be ",
      "repeatable, so this script will not read the clock for you.",
      call. = FALSE
    )
  }
  defaults
}

# crypto2 has renamed these across versions, so resolve by name and fail loudly
# rather than by position. A column that moved would otherwise put volume where
# market cap belongs, and nothing downstream would notice.
first_present <- function(frame, candidates, what) {
  found <- candidates[candidates %in% names(frame)]
  if (length(found) == 0) {
    stop(
      "crypto2 returned no ", what, " column; looked for ",
      paste(candidates, collapse = ", "), ". Got: ",
      paste(names(frame), collapse = ", "),
      call. = FALSE
    )
  }
  found[[1]]
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  start_date <- format(as.Date(args$start), "%Y%m%d")
  end_date <- format(as.Date(args$end), "%Y%m%d")

  message(
    "pulling CoinMarketCap historical listings ", args$start, " to ", args$end,
    " at ", args$interval, " — this is the one-time pull"
  )
  listings <- crypto2::crypto_listings(
    which = "historical",
    convert = "USD",
    start_date = start_date,
    end_date = end_date,
    interval = args$interval,
    quote = TRUE,
    sleep = 1,
    finalWait = FALSE
  )

  panel <- data.frame(
    ts_utc = format(
      as.Date(listings[[first_present(listings, c("ref_cur_timestamp",
                                                  "timestamp",
                                                  "last_updated",
                                                  "date"), "snapshot date")]]),
      "%Y-%m-%d"
    ),
    cmc_id = as.integer(listings[[first_present(listings, c("id", "cmc_id"), "id")]]),
    symbol = as.character(listings$symbol),
    name = as.character(listings$name),
    cmc_rank = as.integer(listings[[first_present(listings, c("cmc_rank", "rank"), "rank")]]),
    price_usd = as.numeric(listings[[first_present(listings, c("price", "USD_price"), "price")]]),
    market_cap_usd = as.numeric(
      listings[[first_present(listings, c("market_cap", "USD_market_cap"), "market cap")]]
    ),
    volume_24h_usd = as.numeric(
      listings[[first_present(listings, c("volume_24h", "USD_volume_24h"), "volume")]]
    ),
    circulating_supply = as.numeric(listings$circulating_supply),
    stringsAsFactors = FALSE
  )
  panel <- panel[order(panel$ts_utc, panel$cmc_id), PANEL_COLUMNS]

  dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(panel, args$out, row.names = FALSE, na = "")
  message(
    "wrote ", nrow(panel), " rows covering ",
    length(unique(panel$cmc_id)), " assets to ", args$out
  )
}

main()
