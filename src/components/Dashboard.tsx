// src/components/Dashboard.tsx
import React from 'react';
import { Container, Grid, Box, Typography, Paper, Button } from '@mui/material';

const Dashboard = () => (
  <Container sx={{ marginTop: '20px' }}>
    <Grid container spacing={4}>
      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ padding: '20px', height: '100%' }}>
          <Typography variant="h5" gutterBottom>
            Upload Data thing
          </Typography>
          <Typography variant="body1">
            Upload Data thing built in here (make it look nice)
          </Typography>
          <Box sx={{ marginTop: '20px' }}>
            <Button variant="contained" color="primary">
              Upload
            </Button>
          </Box>
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ padding: '20px', height: '100%' }}>
          <Typography variant="h5" gutterBottom>
            Brain anatomy
          </Typography>
          <Typography variant="body1">
            Brain anatomy (could be 3d or ni.gz but I'd prefer 3d)
          </Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ padding: '20px', height: '100%' }}>
          <Typography variant="h5" gutterBottom>
            Valence-Arousal/magnitude graph
          </Typography>
          <Typography variant="body1">
            This is more in vein with our current template, kinda like the infrastructure management thing but better
          </Typography>
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper elevation={3} sx={{ padding: '20px', height: '100%' }}>
          <Typography variant="h5" gutterBottom>
            Analysis
          </Typography>
          <Typography variant="body1">
            Analysis: (it would keep scrolling) some written or gpted analysis of the data: I think GPT is a good idea that way we can have like some nice auto generated diagnosis or we can have preset things that work with our inputs
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  </Container>
);

export default Dashboard;
