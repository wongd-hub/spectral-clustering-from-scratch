# Load dependencies, set up graph fonts ----
library(dplyr)
library(mlbench)
library(ggplot2)
library(showtext)
library(jsonlite)
library(mclust)
library(tidyr)
library(purrr)
library(kernlab)
# library(patchwork)

plot_dim <- list(width = 1684, height = 1684 * 9/16)

sysfonts::font_add_google('Space Grotesk', "space_grotesk")

showtext::showtext_auto()


# Create training data ----

set.seed(24601)
spirals_raw <- mlbench::mlbench.spirals(500, cycles = 1, sd = 0.025)

data <- as_tibble(spirals_raw$x, .name_repair = ~c('x', 'y')) %>% 
  mutate(true_label = as.factor(spirals_raw$classes))

data %>% 
  ggplot(aes(x, y, colour = true_label)) +
  geom_point() +
  labs(
    title = 'Spirals dataset', 
    colour = 'True cluster'
  ) +
  theme_minimal(base_family = "space_grotesk") + 
  theme(legend.position = 'bottom')

ggsave(
  file.path('plots', "spirals-data.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)

# Training loop ----

make_affinity_matrix <- function(S, k) {

  n  <- nrow(S)
  AM <- matrix(0, nrow = n, ncol = n)

  # For each observation...
  for ( i in seq_len(n) ) {

    # Find the k-th largest similarity scores
    kth_largest <- sort(S[i, ], decreasing = TRUE)[k]

    # and filter out the rest
    neighbours        <- S[i, ] >= kth_largest
    AM[i, neighbours] <- S[i, neighbours]

    # Keep it symmetrical; keep the edge if either 
    # node considers the other a neighbour
    AM[neighbours, i] <- pmax(AM[neighbours, i], S[i, neighbours])

  }

  AM

}

train_spectral_clustering <- function(
  data, 
  c_mult, k_nn, 
  .centers  = 2,
  transform = function(d, .c_param) exp(-d / .c_param), 
  training  = TRUE,
  .seed     = 24601
) {

  set.seed(.seed)

  # Similarity matrix ----
  c_param   <- median(dist(data %>% select(x, y))) * c_mult

  similarity_matrix <- data %>% 
    select(x, y) %>% 
    dist() %>% 
    as.matrix() %>% 
    transform(c_param)

  # K-nearest neighbour affinity matrix ----
  A <- make_affinity_matrix(similarity_matrix, k = k_nn)

  # Graph Laplacian ----
  D <- diag(rowSums(A))
  L <- D - A

  # Eigendecomposition ----
  eig <- eigen(L)

  # Eigenvalues smallest -> largest for readability
  eigenvalues <- tibble(
    index = seq_along(eig$values),
    value = rev(eig$values)
  )
  
  # K-means on embedding ----
  embedding <- eig$vectors[
    , (ncol(eig$vectors) - .centers + 1):ncol(eig$vectors)
  ] %>% 
    as_tibble(.name_repair = ~c("z1", "z2"))
  
  km <- kmeans(embedding, centers = .centers, nstart = 10)

  if ( training ) {

    tibble(
      c_mult,
      k_nn,
      ari = mclust::adjustedRandIndex(km$cluster, data$true_label)
    )

  } else {

    list(
      eigenvalues = eigenvalues,
      embedding   = embedding,
      km          = km
    )

  }

}



# Rudimentary grid search
c_mult <- c(0.01, 0.1, 0.5, 1, 2)
k_nn   <- c(1, 10, 30, 50, 70, 100)

results <- crossing(
  c_mult, k_nn
) %>% 
  pmap_dfr(train_spectral_clustering, data = data)

results %>% 
  arrange(desc(ari))


output <- train_spectral_clustering(data, c_mult = 0.05, k_nn = 10, training = FALSE)

## Stepped out ----

set.seed(24601)

### Similarity matrix ----
c_param <- median(dist(data %>% select(x, y))) * 0.05

similarity_matrix <- data %>% 
  select(x, y) %>% 
  # Gets distance between each pair of points and returns it as a 
  # lower triangular matrix
  dist() %>% 
  # Turns into a full matrix, and mirrors both sides of the diagonal
  as.matrix() %>% 
  # Runs the Gaussian kernel over all matrix elements
  (function(d, .c_param) exp(-d / .c_param))(c_param)

### K-nearest neighbour affinity matrix ----
A <- make_affinity_matrix(similarity_matrix, k = 10)

all.equal(A, t(A))

#>      [,1] [,2] [,3] [,4] [,5] 
#> [1,] 1.00 0.00 0.00 0.00 0.00 ...
#> [2,] 0.00 1.00 0.00 0.00 0.74
#> [3,] 0.00 0.00 1.00 0.00 0.00
#> [4,] 0.00 0.00 0.00 1.00 0.00
#> [5,] 0.00 0.74 0.00 0.00 1.00
#>      ...                      ⋱

### Graph Laplacian ----
D <- diag(rowSums(A))
L <- D - A

#>      [,1]  [,2] [,3] [,4]  [,5]
#> [1,] 2.37  0.00 0.00 0.00  0.00 ...
#> [2,] 0.00  3.35 0.00 0.00 -0.74
#> [3,] 0.00  0.00 3.39 0.00  0.00
#> [4,] 0.00  0.00 0.00 2.50  0.00
#> [5,] 0.00 -0.74 0.00 0.00  3.47
#>      ...                        ⋱

### Eigendecomposition ----
eig <- eigen(L)

# Eigenvalues smallest -> largest for readability
eigenvalues <- tibble(
  index = seq_along(eig$values),
  value = rev(eig$values)
)

### K-means on embedding ----
embedding <- eig$vectors[
  , (ncol(eig$vectors) - 2 + 1):ncol(eig$vectors)
] %>% 
  as_tibble(.name_repair = ~c("z1", "z2"))

km <- kmeans(embedding, centers = 2, nstart = 10)

eigenvalues %>% 
  ggplot(aes(index, value, color = value)) +
  geom_point() +
  geom_line() +
  coord_cartesian(
    xlim = c(1, 12)
    , ylim = c(0, 0.01)
  ) +
  # annotate(
  #   "text", x = 11.5, y = 0.001, 
  #   label = "...", size = 8, hjust = 0
  # ) +
  geom_hline(yintercept = 0.00075, linetype = 'dotted') +
  labs(
    title = "Eigenvalues of graph Laplacian",
    x = "Index", y = "Eigenvalue",
    caption = 'Dotted line marks the first eigengap'
  ) + 
  theme_minimal(base_family = "space_grotesk") + 
  theme(legend.position = 'none')

ggsave(
  file.path('plots', "smallest-eigenvalues.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)

data %>% 
  ggplot(aes(x, y, colour = as.factor(km$cluster))) +
  geom_point() +
  labs(
    title = "Spectral Clustering results"
  ) + 
  theme_minimal(base_family = "space_grotesk") + 
  theme(legend.position = 'none')

ggsave(
  file.path('plots', "spectral-clustering-results.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)

# Take the k smallest eigenvectors (last k columns from eigen())
embedding %>% 
  ggplot(aes(z1, z2, colour = as.factor(output$km$cluster))) +
  geom_point() +
  labs(
    title = "Data in eigenspace"
  ) + 
  theme_minimal(base_family = "space_grotesk") + 
  theme(legend.position = 'none')

ggsave(
  file.path('plots', "projected-on-eigencoordinates.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)

## Perform out-of-the-box k-means ----

kmeans_model <- kmeans(spirals_raw$x, centers = 2, iter.max = 100)

data %>% 
  mutate(
    pred_label = as.integer(kmeans_model$cluster)
  ) %>% 
  ggplot(aes(x, y, colour = as.factor(pred_label))) +
  geom_point() +
  labs(
    title = "Naive K-means clustering result"
  ) + 
  theme_minimal(base_family = "space_grotesk") + 
  theme(legend.position = 'none')

ggsave(
  file.path('plots', "kmeans-alone.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)

## Perform out-of-the-box spectral clustering ----

specc_model <- specc(spirals_raw$x, centers = 2)

mclust::adjustedRandIndex(
  as.numeric(specc_model),
  spirals_raw$classes
) # instant 1

# manual <- data %>% 
#   ggplot(aes(x, y, colour = as.factor(output$km$cluster))) +
#   geom_point() +
#   theme(legend.position = 'bottom')

# specc_ver <- data %>% 
#   ggplot(aes(x, y, colour = as.factor(as.numeric(specc_model)))) +
#   geom_point() +
#   theme(legend.position = 'bottom')

# manual + specc_ver

bind_rows(
  data %>% 
    mutate(
      model      = 'True Labels',
      pred_label = as.integer(true_label)
    ),
  data %>% 
    mutate(
      model      = 'K-means Clustering',
      pred_label = as.integer(kmeans_model$cluster)
    ),
  data %>% 
    mutate(
      model      = 'Spectral Clustering: from scratch',
      pred_label = as.integer(output$km$cluster)
    ),
  data %>% 
    mutate(
      model      = 'Spectral Clustering: specc()',
      pred_label = as.integer(specc_model)
    )
) %>% 
  ggplot(aes(x, y, colour = as.factor(pred_label))) +
  geom_point() +
  facet_wrap(vars(model)) +
  theme(legend.position = 'none') +
  labs(
    title = 'Comparison of clustering methods on Two Spirals'
  )

ggsave(
  file.path('plots', "all-methods.jpg"),
  device = 'jpg',
  width = plot_dim$width,
  height = plot_dim$height,
  units = 'px',
  dpi = 250
)
