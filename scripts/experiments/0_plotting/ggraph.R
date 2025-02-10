library(ggraph)
library(tidygraph)
library(tidyverse)
library(igraph)


path = "/Users/vcerq/Dropbox/Research/graph_based_time_series_aug/assets/results/plot_data/example.csv"

df <- read.csv(path)

# Create graph of highschool friendships
graph <- as_tbl_graph(df) |>
  mutate(Popularity = centrality_degree(mode = 'in'))

# # plot using ggraph
# ggraph(graph, layout = 'kk') +
#   geom_edge_fan(aes(alpha = after_stat(index)), show.legend = FALSE) +
#   geom_node_point(aes(size = Popularity)) +
#   #facet_edges(~year) +
#   theme_graph(foreground = 'steelblue', fg_text_colour = 'white') +
#   guides(size = "none")

ggraph(graph, layout = 'kk') +
  # Add edges
  geom_edge_fan(
    aes(alpha = after_stat(index)),
    show.legend = FALSE,
    edge_width = 0.5,
    edge_colour = "grey5",
    edge_alpha = 0.3
  ) +
  # Enhance nodes
  geom_node_point(
    aes(size = Popularity),
    color = "white",
    fill = "steelblue",
    shape = 21,
    stroke = 0.5
  ) +
  # Scale node sizes more appropriately
  scale_size_continuous(range = c(3, 10)) +
  # Clean minimal theme with white background
  theme_graph(
    base_family = "serif",
    background = 'white',
    text_colour = "black"
  ) +
  # Remove legends
  guides(size = "none") +
  # Ensure clean white background
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )
